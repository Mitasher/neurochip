"""
Главный прогон эксперимента по SEU-устойчивости: фон-Нейман (МК) vs нейроморф (AltAI).

Результаты РАЗДЕЛЕНЫ на два эксперимента:

  ┌─ Эксперимент 1 — ЛИНЕЙНАЯ АЛГЕБРА ───────────────────────────────────────┐
  │  Алгоритм:  y = W · x   (умножение тернарной матрицы на бинарный вектор). │
  │  Что меряем: насколько один битфлип искажает выходной вектор.            │
  └──────────────────────────────────────────────────────────────────────────┘
  ┌─ Эксперимент 2 — ПЕРЦЕПТРОН (пороговый классификатор) ────────────────────┐
  │  Алгоритм:  class = argmax(W · x)   (однослойный перцептрон).             │
  │  Что меряем: меняет ли один битфлип РЕШЕНИЕ классификатора.               │
  └──────────────────────────────────────────────────────────────────────────┘

ВХОД (одинаковый для обеих платформ и обоих экспериментов):
  * W — тернарная матрица {-1,0,1} формы (OUT_DIM, IN_DIM);
  * x — бинарный вектор {0,1} длины IN_DIM.

ГДЕ исполняется проверка:
  * фон-Нейман  — numpy-модель обычного МК (float32 и int8). Реального
                  кремния МК тут нет — это эталонная «уязвимая» архитектура.
  * нейроморф   — инъекция в синапс выполняется ПРЯМО НА ЭМУЛЯТОРЕ ЧИПА AltAI
                  (golden model): под повреждённые веса пересобирается
                  placed-конфиг и заново исполняется на чипе.
  * самовосстановление мембраны — отдельный блок на numpy-модели LIF
                  (эмулятор чипа в режиме gm не даёт инъекции в мембрану во
                  время счёта), помечен явно как «не на чипе».

Запуск:
    python seu_experiment.py            # нейроморф-синапс на ЧИПЕ, остальное numpy
    python seu_experiment.py --no-chip  # всё на numpy-моделях (быстро, без чипа)
    python seu_experiment.py --altai-validate   # + сверка numpy-модели с чипом
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np

from config import (
    set_global_seed, RANDOM_SEED, N_TRIALS, N_CHIP_SYNAPSE, TRACE_SAMPLES,
    IN_DIM, OUT_DIM, NUM_CLASSES, N_TICKS, LEAK, V_THRESHOLD,
    VN_TARGETS, NM_TARGETS, VN_DTYPES, MEMBRANE_BITS,
    ALTAI_MODEL_PATH, ALTAI_CONFIG_PATH, ALTAI_VALIDATION_SAMPLES,
    RESULTS_DIR, REPORT_JSON_PATH, FATAL_ABS_THRESHOLD,
)
from algorithms import make_problem, linear_transform, threshold_classifier
from platforms.vonneumann import VonNeumannMCU, quantize_weights
from platforms.neuromorph import NeuromorphChip
from fault_injection import (
    sample_vn_fault, sample_nm_fault,
    flip_bit_float32, flip_bit_int, flip_bit_ternary,
)
import metrics


# ============================================================
# Утилиты JSON-безопасного представления выхода
# ============================================================
def _safe_vec(vec) -> list:
    """Вектор → список, с inf/nan как строками (чтобы корректно лечь в JSON)."""
    out = []
    for v in np.asarray(vec, dtype=np.float64).tolist():
        if np.isinf(v):
            out.append('inf' if v > 0 else '-inf')
        elif np.isnan(v):
            out.append('nan')
        else:
            out.append(round(v, 4))
    return out


# ============================================================
# Фон-Нейман: оба эксперимента из общих прогонов
# ============================================================
def _vn_weight_preview(Wq: np.ndarray, dtype: str, row: int, col: int, bit: int):
    """Старое → новое значение слова веса при битфлипе (для трассы)."""
    old = Wq[row, col]
    if dtype == 'float32':
        new = np.float32(flip_bit_float32(float(old), bit))
    else:
        new = np.int8(flip_bit_int(int(old), bit, 8))
    return old, new


def _vn_fault_desc(fault, Wq, dtype) -> str:
    if fault.target == 'weight':
        old, new = _vn_weight_preview(Wq, dtype, fault.row, fault.col, fault.bit)
        return (f"вес W[{fault.row},{fault.col}] ({dtype}), бит {fault.bit}: "
                f"{old} → {new}")
    if fault.target == 'accumulator':
        return (f"регистр-аккумулятор нейрона {fault.row} на шаге k={fault.step}, "
                f"бит {fault.bit}")
    if fault.target == 'pc':
        return (f"счётчик цикла нейрона {fault.row} на шаге k={fault.step}, "
                f"бит {fault.bit} (искажение индекса доступа)")
    return fault.short()


def run_vn_block(problem, dtype, rng) -> dict:
    """
    Один прогон фон-Неймановского МК заданной разрядности. Из КАЖДОЙ эмуляции
    битфлипа получаем сразу оба исхода: вектор W·x (эксп.1) и класс argmax (эксп.2).
    """
    mcu = VonNeumannMCU(problem.W, dtype)
    Wq = quantize_weights(problem.W, dtype)

    # эталон
    y_ref, _ = mcu.linear_transform(problem.x)
    c_ref = int(np.argmax(y_ref))
    active_ref = frozenset(j for j in range(OUT_DIM) if y_ref[j] > 0)

    lin_per, cls_per = {}, {}
    lin_all, cls_all = [], []
    lin_traces, cls_traces = [], []

    for target in VN_TARGETS:
        lin_out, cls_out = [], []
        for i in range(N_TRIALS):
            fault = sample_vn_fault(rng, target, OUT_DIM, IN_DIM, dtype)
            y_f, status = mcu.linear_transform(problem.x, fault)

            lo = metrics.vector_outcome(y_ref, y_f, status)
            # Эксп.2: решение в знаковой форме step(W·x>0) — то же, что считает чип.
            finite = (status == 'ok') and np.all(np.isfinite(y_f))
            active_f = frozenset(j for j in range(OUT_DIM) if y_f[j] > 0) if finite else frozenset()
            co = metrics.decision_outcome(active_ref, active_f, status)
            # argmax — дополнительно (полную величину МК умеет, чип — нет)
            c_f = int(np.argmax(y_f)) if finite else None
            lin_out.append(lo); cls_out.append(co)

            if i < TRACE_SAMPLES:
                desc = _vn_fault_desc(fault, Wq, dtype)
                lin_traces.append({
                    'fault': desc,
                    'before': _safe_vec(y_ref),
                    'after': _safe_vec(y_f),
                    'rel_deviation': (round(lo.rel_deviation, 3)
                                      if np.isfinite(lo.rel_deviation) else 'inf'),
                    'status': status,
                })
                cls_traces.append({
                    'fault': desc,
                    'before_active': sorted(active_ref),
                    'after_active': (sorted(active_f) if finite else 'повреждён (inf/crash)'),
                    'decision_changed': co.class_changed,
                    'argmax_before': c_ref,
                    'argmax_after': c_f,
                    'status': status,
                })
        lin_per[target] = metrics.aggregate(lin_out, 'vector')
        cls_per[target] = metrics.aggregate(cls_out, 'class')
        lin_all.extend(lin_out); cls_all.extend(cls_out)

    return {
        'linear': {
            'output_type': f'вектор логитов W·x ({dtype}), полная разрядность',
            'golden': _safe_vec(y_ref),
            'per_target': lin_per,
            'all_targets': metrics.aggregate(lin_all, 'vector'),
            'traces': lin_traces,
        },
        'perceptron': {
            'output_type': (f'решение step(W·x>0) — знаковый паттерн (сопоставимо с чипом); '
                            f'argmax(W·x)={c_ref} как дополнительный однопобедительный readout'),
            'golden': {'active_set': sorted(active_ref), 'argmax_class': c_ref},
            'per_target': cls_per,
            'all_targets': metrics.aggregate(cls_all, 'class'),
            'traces': cls_traces,
        },
    }


# ============================================================
# Нейроморф: инъекция в синапс ПРЯМО НА ЧИПЕ AltAI
# ============================================================
def run_chip_synapse_block(problem, rng, n_faults) -> dict:
    """
    Прогон инъекций в синапс на РЕАЛЬНОМ эмуляторе чипа AltAI. Каждая инъекция:
    битфлип тернарного веса → пересборка placed-конфига → исполнение на golden
    model → сравнение знакового выхода чипа с эталонным. Возвращает оба
    эксперимента (линейный знаковый вектор и решение-active-set).
    """
    from platforms.altai_chip import AltaiChip
    chip = AltaiChip(problem.W)
    golden = chip.run_clean(problem.x)
    W0 = problem.W.astype(np.int64)

    lin_out, cls_out, traces = [], [], []
    seen, attempts, cache = set(), 0, {}
    while len(traces) < n_faults and attempts < n_faults * 10:
        attempts += 1
        fault = sample_nm_fault(rng, 'synapse', OUT_DIM, IN_DIM, N_TICKS, MEMBRANE_BITS)
        key = (fault.row, fault.col, fault.bit)
        if key in seen:
            continue
        seen.add(key)

        old = int(W0[fault.row, fault.col])
        new = flip_bit_ternary(old, fault.bit)
        Wf = W0.copy(); Wf[fault.row, fault.col] = new

        masked = np.array_equal(Wf, W0)             # битфлип не изменил вес
        stimulated = bool(problem.x[fault.col] == 1)
        if masked:
            res = golden
        else:
            h = Wf.tobytes()
            res = cache.get(h)
            if res is None:
                print(f"    чип: сбой #{len(traces)+1}/{n_faults} — "
                      f"W[{fault.row},{fault.col}] {old}→{new}, пересборка конфига...",
                      flush=True)
                res = chip.run_with_synapse_fault(problem.x, Wf)
                cache[h] = res

        lo = metrics.vector_outcome(golden['binary'], res['binary'], 'ok')
        co = metrics.decision_outcome(golden['active'], res['active'])
        lin_out.append(lo); cls_out.append(co)

        note = []
        if masked:
            note.append("битфлип не изменил тернарный вес (маскирован)")
        elif not stimulated:
            note.append("x=0 на этом входе → синапс не возбуждён (маскирован)")
        traces.append({
            'fault': f"синапс W[{fault.row},{fault.col}] (тернар, 2 бита), "
                     f"бит {fault.bit}: {old} → {new}"
                     + (f"  [{'; '.join(note)}]" if note else ""),
            'input_synapse_stimulated': stimulated,
            'linear_before_spikes': golden['binary'].tolist(),
            'linear_after_spikes': res['binary'].tolist(),
            'linear_rel_deviation': round(lo.rel_deviation, 3),
            'perceptron_before_active': sorted(golden['active']),
            'perceptron_after_active': sorted(res['active']),
            'decision_changed': co.class_changed,
        })

    return {
        'golden': golden,
        'linear': {
            'output_type': 'бинарный знаковый вектор спайков чипа {0,1}^OUT_DIM',
            'golden_spikes': golden['binary'].tolist(),
            'all_targets': metrics.aggregate(lin_out, 'vector'),
            'traces': traces,
        },
        'perceptron': {
            'output_type': 'множество спайкующих нейронов (active set) на чипе',
            'golden_active': sorted(golden['active']),
            'all_targets': metrics.aggregate(cls_out, 'class'),
            'traces': traces,
        },
        'n_faults': len(traces),
    }


def run_chip_synapse_block_numpy(problem, rng, n_faults) -> dict:
    """
    Резерв на случай недоступного чипа (--no-chip): та же инъекция в синапс, но
    на numpy-модели нейроморфа. Знаковый выход берём как sign(W·x).
    """
    W0 = problem.W.astype(np.int64)

    def sign_out(W):
        logits = W @ problem.x.astype(np.int64)
        binary = (logits > 0).astype(np.float64)
        return {'binary': binary, 'active': frozenset(int(j) for j in np.where(logits > 0)[0])}

    golden = sign_out(W0)
    lin_out, cls_out, traces = [], [], []
    seen, attempts = set(), 0
    while len(traces) < n_faults and attempts < n_faults * 10:
        attempts += 1
        fault = sample_nm_fault(rng, 'synapse', OUT_DIM, IN_DIM, N_TICKS, MEMBRANE_BITS)
        key = (fault.row, fault.col, fault.bit)
        if key in seen:
            continue
        seen.add(key)
        old = int(W0[fault.row, fault.col]); new = flip_bit_ternary(old, fault.bit)
        Wf = W0.copy(); Wf[fault.row, fault.col] = new
        res = sign_out(Wf)
        lo = metrics.vector_outcome(golden['binary'], res['binary'], 'ok')
        co = metrics.decision_outcome(golden['active'], res['active'])
        lin_out.append(lo); cls_out.append(co)
        traces.append({
            'fault': f"синапс W[{fault.row},{fault.col}] бит {fault.bit}: {old} → {new}",
            'input_synapse_stimulated': bool(problem.x[fault.col] == 1),
            'linear_before_spikes': golden['binary'].tolist(),
            'linear_after_spikes': res['binary'].tolist(),
            'linear_rel_deviation': round(lo.rel_deviation, 3),
            'perceptron_before_active': sorted(golden['active']),
            'perceptron_after_active': sorted(res['active']),
            'decision_changed': co.class_changed,
        })
    return {
        'golden': golden,
        'linear': {'output_type': 'sign(W·x) ∈ {0,1} (numpy-модель, чип недоступен)',
                   'golden_spikes': golden['binary'].tolist(),
                   'all_targets': metrics.aggregate(lin_out, 'vector'), 'traces': traces},
        'perceptron': {'output_type': 'active set = sign(W·x) (numpy-модель, чип недоступен)',
                       'golden_active': sorted(golden['active']),
                       'all_targets': metrics.aggregate(cls_out, 'class'), 'traces': traces},
        'n_faults': len(traces),
    }


# ============================================================
# Нейроморф: самовосстановление мембраны (numpy-модель, НЕ на чипе)
# ============================================================
def run_membrane_block(problem, rng) -> dict:
    """
    Транзиентный сбой мембранного потенциала и его затухание за счёт утечки.
    Эмулятор чипа (gm) не даёт инъекции в мембрану во время счёта, поэтому это
    демонстрация на numpy-модели LIF — помечена явно.
    """
    chip = NeuromorphChip(problem.W)
    c_ref = chip.threshold_classifier(problem.x)[0]
    outcomes, traces = [], []
    for i in range(N_TRIALS):
        fault = sample_nm_fault(rng, 'membrane', OUT_DIM, IN_DIM, N_TICKS, MEMBRANE_BITS)
        c_f, _ = chip.threshold_classifier(problem.x, fault)
        out = metrics.class_outcome(c_ref, c_f, 'ok')
        out.recovery_ticks = chip.membrane_recovery_ticks(problem.x, fault)
        outcomes.append(out)
        if i < TRACE_SAMPLES:
            traces.append({
                'fault': f"мембрана нейрона {fault.neuron} на тике t={fault.step}, "
                         f"бит {fault.bit} (int{MEMBRANE_BITS}, фикс. точка)",
                'before_class': c_ref,
                'after_class': (None if c_f is None else int(c_f)),
                'decision_changed': out.class_changed,
                'recovery_ticks': out.recovery_ticks,
            })
    return {
        'on_chip': False,
        'note': ('numpy-модель LIF: golden model AltAI в режиме gm не выдаёт '
                 'состояние нейрона во время счёта, поэтому инъекция в мембрану и '
                 'самовосстановление за счёт утечки показаны на модели, НЕ на чипе.'),
        'golden_class': c_ref,
        'stats': metrics.aggregate(outcomes, 'class'),
        'traces': traces,
    }


# ============================================================
# Точка входа
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SEU-устойчивость: фон-Нейман vs AltAI")
    parser.add_argument('--no-chip', action='store_true',
                        help="не запускать чип AltAI: инъекцию в синапс считать на numpy-модели")
    parser.add_argument('--altai-validate', action='store_true',
                        help="дополнительно сверить numpy-модель нейроморфа с чипом на чистых входах")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--out', default=REPORT_JSON_PATH)
    args = parser.parse_args()

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    problem = make_problem(seed=args.seed)

    y_ref = linear_transform(problem.W, problem.x)
    c_ref = threshold_classifier(problem.W, problem.x)

    print("=" * 70)
    print("ЭКСПЕРИМЕНТ SEU: фон-Нейман (МК) vs нейроморф (AltAI)")
    print("=" * 70)
    print("ВХОД (общий для обеих платформ):")
    print(f"  W — тернарная матрица {problem.W.shape} ∈ {{-1,0,1}}")
    print(f"  x — бинарный вектор {problem.x.shape} ∈ {{0,1}} = {problem.x.tolist()}")
    print(f"  эталон  W·x = {y_ref.tolist()}   →   argmax = класс {c_ref}")
    print(f"  битфлипов: numpy-модели N_TRIALS={N_TRIALS}, "
          f"чип AltAI (синапс) N_CHIP_SYNAPSE={N_CHIP_SYNAPSE}\n")

    results = {
        'meta': {
            'seed': args.seed, 'n_trials': N_TRIALS, 'n_chip_synapse': N_CHIP_SYNAPSE,
            'in_dim': IN_DIM, 'out_dim': OUT_DIM, 'num_classes': NUM_CLASSES,
            'n_ticks': N_TICKS, 'leak': LEAK, 'v_threshold': V_THRESHOLD,
            'fatal_abs_threshold': FATAL_ABS_THRESHOLD,
            'vn_targets': list(VN_TARGETS), 'vn_dtypes': list(VN_DTYPES),
            'input': {
                'W': problem.W.astype(int).tolist(),
                'x': problem.x.astype(int).tolist(),
                'description': ('W — тернарные веса {-1,0,1}; x — бинарный вход {0,1}. '
                                'Одинаковы для обоих экспериментов и обеих платформ.'),
            },
            'golden': {
                'logits_Wx': y_ref.tolist(),
                'class_argmax': c_ref,
                'sign_active_set': sorted(int(j) for j in np.where(problem.W @ problem.x > 0)[0]),
            },
        },
    }

    # --- считаем блоки платформ ---
    vn_blocks = {}
    for dtype in VN_DTYPES:
        print(f"[фон-Нейман / {dtype}] numpy-модель: {N_TRIALS} битфлипов × "
              f"{len(VN_TARGETS)} мишеней (weight/accumulator/pc)...")
        vn_blocks[f'vonneumann_{dtype}'] = run_vn_block(problem, dtype, rng)

    if args.no_chip:
        print(f"[нейроморф / синапс] numpy-модель (--no-chip): {N_CHIP_SYNAPSE} битфлипов...")
        chip_block = run_chip_synapse_block_numpy(problem, rng, N_CHIP_SYNAPSE)
        chip_label = 'neuromorph_numpy_synapse'
    else:
        print(f"[нейроморф / синапс] ПРЯМО НА ЧИПЕ AltAI: {N_CHIP_SYNAPSE} битфлипов, "
              f"каждый = пересборка конфига (~18-24 с)...")
        try:
            chip_block = run_chip_synapse_block(problem, rng, N_CHIP_SYNAPSE)
            chip_label = 'neuromorph_altai_chip'
        except Exception as exc:
            print(f"  [!] чип недоступен ({exc}); откат на numpy-модель.")
            chip_block = run_chip_synapse_block_numpy(problem, rng, N_CHIP_SYNAPSE)
            chip_label = 'neuromorph_numpy_synapse'

    print("[нейроморф / мембрана] numpy-модель LIF (самовосстановление, не на чипе)...")
    membrane_block = run_membrane_block(problem, rng)

    on_chip = chip_label == 'neuromorph_altai_chip'

    # --- собираем по экспериментам ---
    results['experiment_1_linear_algebra'] = {
        'title': 'Эксперимент 1 — линейная алгебра',
        'algorithm': 'y = W · x  (умножение тернарной матрицы на бинарный вектор)',
        'measures': 'насколько один битфлип искажает выходной вектор (норма отклонения)',
        'platforms': {
            **{name: {'on_chip': False, **blk['linear']} for name, blk in vn_blocks.items()},
            chip_label: {'on_chip': on_chip, 'target': 'synapse',
                         'n_faults': chip_block['n_faults'], **chip_block['linear']},
        },
    }
    results['experiment_2_perceptron'] = {
        'title': 'Эксперимент 2 — перцептрон (пороговый классификатор)',
        'algorithm': 'class = argmax(W · x); на чипе AltAI — знаковый паттерн спайков',
        'measures': 'меняет ли один битфлип РЕШЕНИЕ классификатора',
        'platforms': {
            **{name: {'on_chip': False, **blk['perceptron']} for name, blk in vn_blocks.items()},
            chip_label: {'on_chip': on_chip, 'target': 'synapse',
                         'n_faults': chip_block['n_faults'], **chip_block['perceptron']},
        },
    }
    results['supplementary_membrane_model'] = membrane_block
    results['altai_validation'] = None

    # --- опциональная сверка numpy-модели с чипом на чистых входах ---
    if args.altai_validate:
        print("\n[AltAI] сверка numpy-модели нейроморфа с чипом на чистых входах...")
        from platforms.neuromorph import validate_against_altai
        val_rng = np.random.default_rng(args.seed + 1)
        xs = val_rng.integers(0, 2, size=(ALTAI_VALIDATION_SAMPLES, IN_DIM))
        val = validate_against_altai(problem.W, xs, ALTAI_MODEL_PATH, ALTAI_CONFIG_PATH, N_TICKS)
        results['altai_validation'] = val
        if val:
            print(f"[AltAI] согласие по спайковой активности: "
                  f"{val['active_set_agreement']*100:.1f}% из {val['samples']} входов")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nСырые результаты сохранены: {args.out}")

    _print_headline(results, chip_label, on_chip)
    return results


def _print_headline(results, chip_label, on_chip):
    chip_where = "ЧИП AltAI" if on_chip else "numpy-модель"
    print("\n" + "=" * 70)
    print("СВОДКА ПО ЭКСПЕРИМЕНТАМ")
    print("=" * 70)

    e1 = results['experiment_1_linear_algebra']['platforms']
    print("\nЭксперимент 1 (линейная алгебра): макс. отн. отклонение выхода ‖Δ‖/‖ref‖")
    for name, blk in e1.items():
        s = blk['all_targets']
        where = "чип" if blk.get('on_chip') else "numpy"
        print(f"  {name:<26} [{where:>5}]  макс={s.get('max_rel_deviation', 0):>10.2g}×  "
              f"inf-выходов={s.get('inf_deviation_rate', 0)*100:>5.1f}%")

    e2 = results['experiment_2_perceptron']['platforms']
    print("\nЭксперимент 2 (перцептрон): частота смены решения классификатора")
    for name, blk in e2.items():
        s = blk['all_targets']
        where = "чип" if blk.get('on_chip') else "numpy"
        print(f"  {name:<26} [{where:>5}]  смена решения={s['class_change_rate']*100:>5.1f}%  "
              f"fatal={s['fatal_rate']*100:>5.1f}%  crash={s['crash_rate']*100:>5.1f}%")

    mem = results['supplementary_membrane_model']['stats']
    if mem.get('recovery_median_ticks') is not None:
        print(f"\nМембрана (numpy-модель, не на чипе): самовосстановление за "
              f"{mem['recovery_median_ticks']:.0f} тиков (медиана), "
              f"восстановилось {mem.get('recovery_rate', 0)*100:.0f}%.")

    print(f"\nИнъекция в синапс нейроморфа выполнена на: {chip_where}.")
    print("Полный анализ и графики — python report.py  →  results/ и docs/03_experiment.md")


if __name__ == "__main__":
    main()
