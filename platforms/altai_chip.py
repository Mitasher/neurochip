"""
Запуск тернарного слоя на РЕАЛЬНОМ эмуляторе чипа AltAI (golden model).

В отличие от numpy-модели (`platforms/neuromorph.py`), здесь и эталонный
прогон, и каждый прогон со сбоем выполняются driver'ом AltAI
(`inference_type='gm'`) — то есть «проверка происходит прямо на эмуляторе чипа».

Как чип отвечает на один тернарный слой `TernaryDense + Heaviside`:
  * каждый выходной нейрон выдаёт РОВНО ОДИН спайк, если его ток (W·x)_j > 0,
    и не спайкует иначе (порог знака). Это проверено: множество спайкующих
    нейронов = sign(W·x) (см. validate_against_altai в neuromorph.py).
  * значение спайка в `get_spikes()` — это ИНДЕКС спайкнувшего нейрона
    (метка класса), пустые ячейки помечены большим отрицательным сентинелом.

Поэтому НАТИВНЫЙ выход чипа — это бинарный знаковый вектор b ∈ {0,1}^OUT_DIM
(b_j = 1 ⇔ нейрон j спайкнул ⇔ (W·x)_j > 0). Из одного прогона чипа получаем:
  * для эксперимента 1 (линейная алгебра): сам вектор b (амплитуда ограничена 1);
  * для эксперимента 2 (перцептрон): решение = множество спайкующих нейронов.

ВАЖНО (без приукрашивания): одиночный слой на чипе — это пороговый детектор
ЗНАКА, он не ранжирует выходы по величине, поэтому «класс = argmax по величине»
на этом чипе физически не считывается. Нативное решение перцептрона на чипе —
это знаковый паттерн (active set). Именно его смену мы и измеряем как
повреждение решения классификатора.

Инъекция сбоя в синапс выполняется честно «в железе»: тернарный вес после
битфлипа кладётся в матрицу, конфиг чипа ПЕРЕСОБИРАЕТСЯ Placer'ом под новые веса
и снова исполняется на golden model. Прицельная правка одного синапса на уже
собранном чипе невозможна: Placer разносит MAC одного слоя по сетке ядер
(здесь 32×4) с непрозрачной раскладкой синапсов — поэтому единственный
достоверный способ внести сбой в вес на чипе — пересборка под новую матрицу.
"""

from __future__ import annotations

import os
import sys
import contextlib
import numpy as np

from config import N_TICKS, ALTAI_CACHE_DIR


# Сентинел «нет спайка» в get_spikes() — большое отрицательное число.
_NO_SPIKE_BELOW = 0


@contextlib.contextmanager
def _silenced():
    """Заглушить шумный C-level вывод Placer'а/driver'а (Active core..., tqdm)."""
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(old_out, 1); os.dup2(old_err, 2)
            os.close(old_out); os.close(old_err)


# ============================================================
# Сборка placed-конфига чипа под конкретную весовую матрицу
# ============================================================
def build_chip_config(W: np.ndarray, model_path: str, config_path: str,
                      quiet: bool = True) -> bool:
    """
    Собрать placed-конфиг тернарного слоя `TernaryDense+Heaviside` под веса W
    через Placer. Возвращает True, если конфиг готов (существует на диске).
    Кэшируется: если config_path уже есть — пересборки нет.
    """
    if os.path.exists(config_path):
        return True
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    try:
        import tensorflow as tf  # noqa: F401
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

    ctx = _silenced() if quiet else contextlib.nullcontext()
    with ctx:
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        out_dim, in_dim = W.shape
        inp = layers.Input(shape=(in_dim,))
        out = TernaryDense(out_dim, activation=heaviside, use_bias=False)(inp)
        model = models.Model(inp, out)
        weights = model.get_weights()
        Wt = W.T.astype(np.float32)              # keras kernel: (in, out)
        weights[0] = Wt
        if len(weights) > 1:
            weights[1] = Wt                      # kernel_prev (кэш квантизации)
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


def _weights_hash(W: np.ndarray) -> str:
    """Короткий стабильный хэш тернарной матрицы для имени конфига."""
    b = np.ascontiguousarray(np.round(W).astype(np.int8)).tobytes()
    import hashlib
    return hashlib.sha1(b).hexdigest()[:12]


# ============================================================
# Чип AltAI
# ============================================================
class AltaiChip:
    """
    Обёртка над golden model AltAI.

    КАЖДЫЙ прогон (эталон и любой сбой) исполняется в ОТДЕЛЬНОМ процессе-воркере
    (`platforms/altai_worker.py`). Причина: golden model держит ГЛОБАЛЬНОЕ
    состояние сетки ядер, которое не сбрасывается между повторными build() в
    одном процессе — после ~2 сборок устройство перестаёт выдавать спайки
    (проверено: golden и первый сбой корректны, второй сбой → пустой выход).
    Свежий процесс = чистое устройство = достоверный результат каждой инъекции.
    """

    def __init__(self, W: np.ndarray, n_ticks: int = N_TICKS):
        self.W = np.round(W).astype(np.int64)
        self.out_dim, self.in_dim = self.W.shape
        self.n_ticks = n_ticks
        self._dir = os.path.join(ALTAI_CACHE_DIR, 'worker')
        os.makedirs(self._dir, exist_ok=True)
        self._x_path = os.path.join(self._dir, 'x.npy')
        self._root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # заранее построим/прогреем эталонный конфиг в кэше (быстрый повторный билд)
        gm = os.path.join(ALTAI_CACHE_DIR, 'ternary_layer.h5')
        gc = os.path.join(ALTAI_CACHE_DIR, 'ternary_layer_snn.json')
        if not build_chip_config(self.W, gm, gc):
            raise RuntimeError("AltAI: не удалось собрать эталонный конфиг чипа")

    # --------------------------------------------------------
    def _run_worker(self, W: np.ndarray, x: np.ndarray) -> dict:
        """Собрать конфиг под веса W и исполнить вход x на чипе в свежем процессе."""
        import subprocess, json
        w_path = os.path.join(self._dir, f'W_{_weights_hash(W)}.npy')
        np.save(w_path, np.round(W).astype(np.int64))
        np.save(self._x_path, x.astype(np.int32))
        try:
            proc = subprocess.run(
                [sys.executable, '-m', 'platforms.altai_worker',
                 w_path, self._x_path, str(self.n_ticks)],
                cwd=self._root, capture_output=True, text=True, timeout=180,
            )
        finally:
            if os.path.exists(w_path):
                os.remove(w_path)
        line = None
        for ln in proc.stdout.splitlines():
            s = ln.strip()
            if s.startswith('{') and s.endswith('}'):
                line = s
        if line is None:
            raise RuntimeError(f"AltAI worker не вернул результат:\n{proc.stdout[-500:]}\n{proc.stderr[-500:]}")
        d = json.loads(line)
        if 'error' in d:
            raise RuntimeError(f"AltAI worker: {d['error']}")
        binary = np.asarray(d['binary'], dtype=np.float64)
        active = frozenset(int(j) for j in d['active'])
        return {'binary': binary, 'active': active}

    def run_clean(self, x: np.ndarray) -> dict:
        """Эталонный прогон (без сбоя) на чипе."""
        return self._run_worker(self.W, x)

    def run_with_synapse_fault(self, x: np.ndarray, W_faulted: np.ndarray) -> dict:
        """Исполнить вход x на чипе с повреждёнными весами W_faulted (свежий процесс)."""
        return self._run_worker(W_faulted, x)
