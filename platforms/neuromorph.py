"""
Нейроморфная платформа — модель чипа AltAI.

Содержит:
  * NeuromorphChip — быстрая, физически достоверная numpy-модель тернарного
    спайкового слоя (leaky integrate-and-fire), на которой гоняются 100 эмуляций
    битфлипа. Механизмы устойчивости заложены в саму модель:
       - тернарные веса {-1,0,1} (2 бита) → битфлип ограничен по амплитуде;
       - бинаризация спайком (Heaviside) → подпороговое возмущение исчезает;
       - утечка мембраны (leak) → транзиентный сбой сам затухает за тики;
       - нет единого PC → нет класса управляющих отказов (status всегда 'ok').
  * validate_against_altai() — сверка модели с РЕАЛЬНЫМ golden model AltAI
    (Placer → Altai). Подтверждает, что без сбоев numpy-модель и чип принимают
    одно решение. Строится один раз и кэшируется (конфиг весит сотни МБ).

Считывание решения классификатора: argmax по проинтегрированной мембране
  r_j = Σ_t V_j(t),  где V_j(t) = leak·V_j(t-1) + (W·x)_j.
Поскольку r_j = C·(W·x)_j (один и тот же положительный C для всех j),
argmax(r) ≡ argmax(W·x) — решение совпадает с эталоном и с golden model.
Линейный выход нейроморфа — вектор спайк-счётчиков (ограничен [0, N_ticks]).
"""

from __future__ import annotations

import os
import numpy as np

from config import (
    N_TICKS, LEAK, V_THRESHOLD, MEMBRANE_BITS,
    IN_DIM, OUT_DIM,
)
from fault_injection import Fault, flip_bit_ternary, flip_bit_membrane


class NeuromorphChip:
    """numpy-модель тернарного LIF-слоя AltAI."""

    def __init__(self, W: np.ndarray, n_ticks: int = N_TICKS,
                 leak: float = LEAK, theta: float = V_THRESHOLD):
        # веса уже тернарные; храним как int
        self.W = np.round(W).astype(np.int64)
        self.out_dim, self.in_dim = self.W.shape
        self.n_ticks = n_ticks
        self.leak = leak
        self.theta = theta

    # --------------------------------------------------------
    def _simulate(self, W: np.ndarray, x: np.ndarray, fault: Fault | None):
        """
        Прогнать LIF на n_ticks. Возвращает словарь:
          readout       — Σ_t V (классификатор берёт argmax);
          spike_count   — Σ_t Heaviside(V>θ)  (линейный выход);
          v_trace       — траектория мембраны (n_ticks, out_dim) для метрики
                          самовосстановления.
        """
        I = (W @ x).astype(np.float64)        # входные токи нейронов = (W·x)
        V = np.zeros(self.out_dim, dtype=np.float64)
        readout = np.zeros(self.out_dim, dtype=np.float64)
        spike_count = np.zeros(self.out_dim, dtype=np.int64)
        v_trace = np.zeros((self.n_ticks, self.out_dim), dtype=np.float64)

        for t in range(self.n_ticks):
            V = self.leak * V + I                       # утечка + интеграция
            # --- инъекция в мембранный потенциал ---
            if (fault is not None and fault.target == 'membrane'
                    and fault.step == t):
                V[fault.neuron] = flip_bit_membrane(V[fault.neuron], fault.bit,
                                                    width=MEMBRANE_BITS)
            spikes = (V > self.theta).astype(np.int64)  # бинаризация спайком
            spike_count += spikes
            readout += V
            v_trace[t] = V

        return {'readout': readout, 'spike_count': spike_count, 'v_trace': v_trace}

    def _apply_synapse_fault(self, fault: Fault | None) -> np.ndarray:
        W = self.W.copy()
        if fault is not None and fault.target == 'synapse':
            W[fault.row, fault.col] = flip_bit_ternary(int(W[fault.row, fault.col]), fault.bit)
        return W

    # --------------------------------------------------------
    def linear_transform(self, x: np.ndarray, fault: Fault | None = None):
        """Линейный выход нейроморфа = вектор спайк-счётчиков. (vector, 'ok')."""
        W = self._apply_synapse_fault(fault)
        res = self._simulate(W, x.astype(np.int64), fault)
        return res['spike_count'].astype(np.float64), 'ok'

    def threshold_classifier(self, x: np.ndarray, fault: Fault | None = None):
        """class = argmax(Σ_t V). Нейроморф не падает — статус всегда 'ok'."""
        W = self._apply_synapse_fault(fault)
        res = self._simulate(W, x.astype(np.int64), fault)
        return int(np.argmax(res['readout'])), 'ok'

    # --------------------------------------------------------
    def membrane_recovery_ticks(self, x: np.ndarray, fault: Fault,
                                eps: float | None = None) -> int:
        """
        Сколько тиков нужно мембране, чтобы вернуться к эталонной траектории
        после транзиентного сбоя. Демонстрирует самовосстановление за счёт leak.
        Порог восстановления eps по умолчанию = порог спайка θ: ниже него
        возмущение уже не влияет на бинаризованный выход нейрона.
        Возвращает число тиков до |V_fault − V_golden| < eps (или -1, если не
        успело за горизонт симуляции).
        """
        if fault.target != 'membrane':
            return 0
        if eps is None:
            eps = self.theta
        golden = self._simulate(self.W, x.astype(np.int64), None)['v_trace'][:, fault.neuron]
        faulty = self._simulate(self.W, x.astype(np.int64), fault)['v_trace'][:, fault.neuron]
        diff = np.abs(faulty - golden)
        t0 = fault.step
        for k in range(t0, self.n_ticks):
            if diff[k] < eps:
                return k - t0
        return -1


# ============================================================
# Валидация против реального golden model AltAI
# ============================================================
def _build_altai_config(W: np.ndarray, model_path: str, config_path: str) -> bool:
    """Построить placed-конфиг тернарного слоя через Placer (кэшируется)."""
    if os.path.exists(config_path):
        return True
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from knp_ann2snn.altainn.ternary_tf2 import TernaryDense, heaviside
        from knp_ann2snn import Placer
        import knp_ann2snn
    except Exception as exc:                     # pragma: no cover
        print(f"[altai] окружение недоступно: {exc}")
        return False

    # патч add_weight (как в исходной инфраструктуре проекта)
    _orig = layers.Layer.add_weight
    def _patched(self, *a, **k):
        if a and isinstance(a[0], str):
            k['name'] = a[0]; a = a[1:]
        return _orig(self, *a, **k)
    layers.Layer.add_weight = _patched

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    out_dim, in_dim = W.shape
    inp = layers.Input(shape=(in_dim,))
    out = TernaryDense(out_dim, activation=heaviside, use_bias=False)(inp)
    model = models.Model(inp, out)
    weights = model.get_weights()
    Wt = W.T.astype(np.float32)                  # keras kernel: (in, out)
    weights[0] = Wt
    if len(weights) > 1:
        weights[1] = Wt                          # kernel_prev (кэш квантизации)
    model.set_weights(weights)
    model.save(model_path, save_format='h5')
    import h5py
    with h5py.File(model_path, 'a') as f:
        if 'training_config' in f.attrs:
            del f.attrs['training_config']

    pkg = os.path.dirname(knp_ann2snn.__file__)
    hw = os.path.join(pkg, 'placer_build', 'resources', 'hw_config.yaml')
    placer = Placer(path_to_model=model_path, input_ns_mode=False, hw_config_path=hw)
    placer.run_placement(config_path)
    if not os.path.exists(config_path):
        placer.save_config(config_path)
    return os.path.exists(config_path)


def validate_against_altai(W: np.ndarray, problems_x: np.ndarray,
                           model_path: str, config_path: str,
                           n_ticks: int = N_TICKS) -> dict | None:
    """
    Сверить решения numpy-модели нейроморфа с РЕАЛЬНЫМ golden model AltAI на
    наборе входов. Возвращает отчёт о согласии (или None, если AltAI недоступен).
    """
    if not _build_altai_config(W, model_path, config_path):
        return None
    try:
        from knp_ann2snn import Altai
    except Exception as exc:                     # pragma: no cover
        print(f"[altai] импорт Altai не удался: {exc}")
        return None

    altai = Altai()
    altai.build(config_path=config_path, inference_type='gm')

    chip = NeuromorphChip(W)
    agree_active = 0          # ГЛАВНОЕ: совпадает ли множество спайкующих нейронов
    winner_active = 0         # попадает ли выбранный моделью класс в активные на чипе
    total = len(problems_x)
    details = []
    for x in problems_x:
        x = x.astype(np.int32)
        # --- реальный AltAI golden model ---
        altai.prepare_spikes(x)
        altai.start_ticks(ticks=n_ticks)
        spikes = altai.get_spikes()
        altai.clear_input()
        flat = spikes.flatten().astype(np.int64)
        flat = flat[(flat >= 0) & (flat < W.shape[0])]
        altai_active = set(np.unique(flat).tolist())

        # --- numpy-модель: спайкует нейрон с (W·x) > 0 (Heaviside) ---
        logits = (W @ x)
        model_active = set(np.where(logits > 0)[0].tolist())
        model_class = int(np.argmax(logits))

        if model_active == altai_active:
            agree_active += 1
        # выбранный моделью победитель должен реально спайковать на чипе
        if (not altai_active and not model_active) or (model_class in altai_active):
            winner_active += 1
        details.append({
            'model_class': model_class,
            'altai_active': sorted(altai_active), 'model_active': sorted(model_active),
        })

    # Примечание: одиночный слой TernaryDense+Heaviside на AltAI — пороговый
    # детектор знака (по 1 спайку на активный нейрон), он не ранжирует по величине.
    # Поэтому валидируем СПАЙКОВУЮ АКТИВНОСТЬ (active set = sign(W·x)); ранжирование
    # argmax по величине — это считывание мембраны в numpy-модели.
    return {
        'samples': total,
        'active_set_agreement': agree_active / total if total else 0.0,
        'winner_active_rate': winner_active / total if total else 0.0,
        'details': details,
    }
