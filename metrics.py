"""
Метрики устойчивости к SEU.

Сравнивают сбойный выход с эталонным (golden) и агрегируют по 100 эмуляциям:
  * отклонение выхода ‖y_fault − y_ref‖ (абсолютное и относительное);
  * доля SDC (Silent Data Corruption) — выход изменился, но программа не упала;
  * частота смены класса (для порогового классификатора) — главный показатель;
  * доля фатальных сбоев (inf/огромное число на МК);
  * доля крахов потока управления (только МК);
  * самовосстановление мембраны (только нейроморф).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ============================================================
# Результат одной эмуляции
# ============================================================
@dataclass
class TrialOutcome:
    status: str                  # 'ok' | 'fatal' | 'crash'
    deviation: float = 0.0       # ‖y_fault − y_ref‖ (абсолютное)
    rel_deviation: float = 0.0   # ‖Δ‖ / (‖y_ref‖ + eps)
    class_changed: bool = False  # сменился ли класс (для классификатора)
    recovery_ticks: int = 0      # тиков до восстановления (нейроморф, membrane)


def vector_outcome(y_ref: np.ndarray, y_fault: np.ndarray, status: str) -> TrialOutcome:
    """Метрика для алгоритма с векторным выходом (линейное преобразование)."""
    if status in ('fatal', 'crash') or y_fault is None or not np.all(np.isfinite(y_fault)):
        return TrialOutcome(status=status if status != 'ok' else 'fatal',
                            deviation=float('inf'), rel_deviation=float('inf'))
    dev = float(np.linalg.norm(y_fault - y_ref))
    ref_norm = float(np.linalg.norm(y_ref)) + 1e-12
    return TrialOutcome(status=status, deviation=dev, rel_deviation=dev / ref_norm,
                        class_changed=False)


def class_outcome(c_ref: int, c_fault, status: str) -> TrialOutcome:
    """Метрика для порогового классификатора (выход — индекс класса)."""
    if status in ('fatal', 'crash') or c_fault is None:
        return TrialOutcome(status=status, class_changed=True)
    return TrialOutcome(status=status, class_changed=(int(c_fault) != int(c_ref)))


def decision_outcome(active_ref, active_fault, status: str = 'ok') -> TrialOutcome:
    """
    Метрика решения перцептрона на РЕАЛЬНОМ чипе AltAI.

    Нативное решение одиночного слоя `TernaryDense+Heaviside` на чипе — это
    множество спайкующих нейронов (знаковый паттерн active set = sign(W·x)).
    Повреждением решения считаем любое изменение этого множества.
    """
    if status in ('fatal', 'crash'):
        return TrialOutcome(status=status, class_changed=True)
    return TrialOutcome(status=status,
                        class_changed=(frozenset(active_fault) != frozenset(active_ref)))


# ============================================================
# Агрегация по N эмуляциям
# ============================================================
def aggregate(outcomes: list[TrialOutcome], output_kind: str) -> dict:
    """
    Свернуть список исходов в сводные показатели.
    output_kind: 'vector' | 'class'.
    """
    n = len(outcomes)
    if n == 0:
        return {}

    n_fatal = sum(o.status == 'fatal' for o in outcomes)
    n_crash = sum(o.status == 'crash' for o in outcomes)
    n_ok = sum(o.status == 'ok' for o in outcomes)

    # SDC — «молчаливое» искажение: программа не упала (ok), но выход изменился.
    if output_kind == 'class':
        n_sdc = sum(o.status == 'ok' and o.class_changed for o in outcomes)
    else:
        n_sdc = sum(o.status == 'ok' and o.deviation > 1e-9 for o in outcomes)

    summary = {
        'n_trials': n,
        'fatal_rate': n_fatal / n,
        'crash_rate': n_crash / n,
        'ok_rate': n_ok / n,
        'sdc_rate': n_sdc / n,
    }

    if output_kind == 'class':
        # сменой класса считаем и явный сбой (fatal/crash), и молчаливую смену
        n_class_changed = sum(o.class_changed or o.status in ('fatal', 'crash')
                              for o in outcomes)
        summary['class_change_rate'] = n_class_changed / n
    else:
        finite = [o for o in outcomes if np.isfinite(o.deviation)]
        devs = np.array([o.deviation for o in finite]) if finite else np.array([])
        rels = np.array([o.rel_deviation for o in finite]) if finite else np.array([])
        summary['mean_deviation'] = float(devs.mean()) if devs.size else 0.0
        summary['max_deviation'] = float(devs.max()) if devs.size else 0.0
        summary['mean_rel_deviation'] = float(rels.mean()) if rels.size else 0.0
        summary['max_rel_deviation'] = float(rels.max()) if rels.size else 0.0
        # доля «взорвавшихся» (нефинитных) выходов
        summary['inf_deviation_rate'] = (n - len(finite)) / n

    # самовосстановление (только там, где есть recovery_ticks > 0)
    rec = [o.recovery_ticks for o in outcomes if o.recovery_ticks > 0]
    censored = sum(o.recovery_ticks == -1 for o in outcomes)
    if rec or censored:
        summary['recovery_median_ticks'] = float(np.median(rec)) if rec else None
        summary['recovery_max_ticks'] = int(np.max(rec)) if rec else None
        summary['recovery_rate'] = len(rec) / (len(rec) + censored) if (len(rec) + censored) else 0.0

    return summary
