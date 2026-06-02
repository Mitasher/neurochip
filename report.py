"""
Отчётность по эксперименту SEU: текстовые таблицы + графики.

Читает results/seu_report.json (создаётся seu_experiment.py). Результаты
разделены на два эксперимента:
  * Эксперимент 1 — линейная алгебра  (y = W·x);
  * Эксперимент 2 — перцептрон         (class = argmax(W·x)).

Строит:
  * results/fig_exp2_decision_change.png — смена решения классификатора (эксп.2);
  * results/fig_exp2_failure_modes.png   — fatal / crash / inf по платформам;
  * results/fig_exp1_deviation.png       — макс. отклонение линейного выхода (эксп.1);
  * results/fig_recovery.png             — самовосстановление мембраны (numpy-модель);
и печатает итоговые таблицы по каждому эксперименту.

Запуск:
    python report.py
"""

from __future__ import annotations

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import REPORT_JSON_PATH, RESULTS_DIR


def _label(name: str) -> str:
    return {
        'vonneumann_float32': 'фон-Нейман\nfloat32',
        'vonneumann_int8': 'фон-Нейман\nint8',
        'neuromorph_altai_chip': 'нейроморф\nЧИП AltAI',
        'neuromorph_numpy_synapse': 'нейроморф\n(numpy)',
    }.get(name, name)


def _color(name: str) -> str:
    if name.startswith('vonneumann_float32'):
        return '#d9772b'
    if name.startswith('vonneumann_int8'):
        return '#c0392b'
    return '#2980b9'


def load(path: str = REPORT_JSON_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# Графики
# ============================================================
def fig_decision_change(results, out):
    plats = results['experiment_2_perceptron']['platforms']
    names = list(plats)
    vals = [plats[n]['all_targets']['class_change_rate'] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([_label(n) for n in names], vals, color=[_color(n) for n in names])
    ax.set_ylabel('Частота смены решения, %')
    ax.set_title('Эксперимент 2 (перцептрон): смена решения от одного битфлипа\n'
                 '(нейроморф — на эмуляторе чипа AltAI)')
    for b, v, n in zip(bars, vals, names):
        tag = v + 0.5
        ax.text(b.get_x() + b.get_width() / 2, tag, f'{v:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25 + 1)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_failure_modes(results, out):
    e2 = results['experiment_2_perceptron']['platforms']
    e1 = results['experiment_1_linear_algebra']['platforms']
    names = list(e2)
    fatal = [e2[n]['all_targets']['fatal_rate'] * 100 for n in names]
    crash = [e2[n]['all_targets']['crash_rate'] * 100 for n in names]
    infr = [e1[n]['all_targets'].get('inf_deviation_rate', 0) * 100 for n in names]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w, fatal, w, label='fatal (inf/огромное)', color='#c0392b')
    ax.bar(x, crash, w, label='crash (поток управления)', color='#7f8c8d')
    ax.bar(x + w, infr, w, label='inf-выход (эксп.1)', color='#e67e22')
    ax.set_xticks(x); ax.set_xticklabels([_label(n) for n in names])
    ax.set_ylabel('Доля прогонов, %')
    ax.set_title('Катастрофические режимы отказа\n(у нейроморфа отсутствуют по построению)')
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_deviation(results, out):
    plats = results['experiment_1_linear_algebra']['platforms']
    names = list(plats)
    maxdev = [max(plats[n]['all_targets'].get('max_rel_deviation', 0), 1e-3) for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([_label(n) for n in names], maxdev, color=[_color(n) for n in names])
    ax.set_yscale('log')
    ax.set_ylabel('Макс. относительное отклонение ‖Δ‖/‖ref‖ (лог)')
    ax.set_title('Эксперимент 1 (линейная алгебра): амплитуда искажения выхода\n'
                 '(широкое слово МК взрывается, знаковый выход чипа ограничен)')
    for b, v in zip(bars, maxdev):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f'{v:.2g}×',
                ha='center', va='bottom', fontweight='bold')
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_recovery(results, out):
    mem = results.get('supplementary_membrane_model', {}).get('stats', {})
    if mem.get('recovery_median_ticks') is None:
        return False
    leak = results['meta']['leak']; theta = results['meta']['v_threshold']
    k = np.arange(0, results['meta']['n_ticks'])
    fig, ax = plt.subplots(figsize=(8, 5))
    for delta0, lbl in [(128.0, 'сбой старшего бита (Δ≈128)'),
                        (16.0, 'сбой среднего бита (Δ≈16)'),
                        (2.0, 'сбой младшего бита (Δ≈2)')]:
        ax.plot(k, delta0 * leak ** k, marker='o', ms=3, label=lbl)
    ax.axhline(theta, color='k', ls='--', lw=1, label=f'порог спайка θ={theta}')
    ax.set_yscale('log'); ax.set_xlabel('Тиков после сбоя')
    ax.set_ylabel('|V_fault − V_golden| (лог)')
    ax.set_title('Самовосстановление мембраны за счёт утечки (numpy-модель, не на чипе)\n'
                 f'медиана = {mem["recovery_median_ticks"]:.0f} тиков, '
                 f'восстановилось {mem.get("recovery_rate", 0)*100:.0f}%')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return True


# ============================================================
# Текстовая сводка
# ============================================================
def print_tables(results):
    m = results['meta']
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ: SEU-устойчивость (фон-Нейман vs нейроморф AltAI)")
    print("=" * 80)
    print(f"ВХОД: W{tuple([m['out_dim'], m['in_dim']])} тернар, "
          f"x{tuple([m['in_dim']])} бинарный = {m['input']['x']}")
    print(f"      эталон W·x = {m['golden']['logits_Wx']}  →  класс {m['golden']['class_argmax']}, "
          f"спайкуют нейроны {m['golden']['sign_active_set']}")
    if results.get('altai_validation'):
        v = results['altai_validation']
        print(f"      сверка numpy-модели с чипом: спайковая активность совпадает на "
              f"{v['active_set_agreement']*100:.1f}% из {v['samples']} входов")

    # --- Эксперимент 1 ---
    e1 = results['experiment_1_linear_algebra']
    print(f"\n--- {e1['title']}:  {e1['algorithm']} ---")
    print(f"  {'платформа':<26}{'где':>8}{'max_rel_dev':>14}{'mean_rel_dev':>14}{'inf-выход':>11}")
    for n, blk in e1['platforms'].items():
        s = blk['all_targets']; where = 'ЧИП' if blk.get('on_chip') else 'numpy'
        print(f"  {n:<26}{where:>8}{s.get('max_rel_deviation', 0):>13.2g}×"
              f"{s.get('mean_rel_deviation', 0):>13.2g}×{s.get('inf_deviation_rate', 0)*100:>10.1f}%")

    # --- Эксперимент 2 ---
    e2 = results['experiment_2_perceptron']
    print(f"\n--- {e2['title']}:  {e2['algorithm']} ---")
    print(f"  {'платформа':<26}{'где':>8}{'смена реш.':>12}{'sdc':>8}{'fatal':>8}{'crash':>8}")
    for n, blk in e2['platforms'].items():
        s = blk['all_targets']; where = 'ЧИП' if blk.get('on_chip') else 'numpy'
        print(f"  {n:<26}{where:>8}{s['class_change_rate']*100:>11.1f}%{s['sdc_rate']*100:>7.1f}%"
              f"{s['fatal_rate']*100:>7.1f}%{s['crash_rate']*100:>7.1f}%")

    # --- Мембрана ---
    mem = results.get('supplementary_membrane_model', {})
    s = mem.get('stats', {})
    if s.get('recovery_median_ticks') is not None:
        print(f"\n--- Дополнительно: самовосстановление мембраны (numpy-модель, НЕ на чипе) ---")
        print(f"  смена решения {s['class_change_rate']*100:.1f}%; восстановление: медиана "
              f"{s['recovery_median_ticks']:.0f} тиков, вернулось {s.get('recovery_rate', 0)*100:.0f}%")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = load()
    print_tables(results)

    made = []
    for name, fn in [('fig_exp2_decision_change.png', fig_decision_change),
                     ('fig_exp2_failure_modes.png', fig_failure_modes),
                     ('fig_exp1_deviation.png', fig_deviation)]:
        path = os.path.join(RESULTS_DIR, name)
        fn(results, path); made.append(path)
    rec = os.path.join(RESULTS_DIR, 'fig_recovery.png')
    if fig_recovery(results, rec):
        made.append(rec)

    print("\nГрафики сохранены:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
