"""
Эталонные алгоритмы эксперимента и генератор задачи.

Два простейших алгоритма (одинаковая семантика на обеих платформах):

  1. Линейное преобразование    y = W · x         (матрица на вектор)
  2. Пороговый классификатор     class = argmax(W · x)   (однослойный перцептрон;
     на нейроморфе порог реализуется спайком по Heaviside)

Веса W — тернарные {-1, 0, 1}: это «родной» формат AltAI и при этом совершенно
корректная схема для обычного МК (там она просто хранится в float32 / int8).
Контраст в эксперименте — не в семантике алгоритма, а в том, как однобитовый
сбой искажает ШИРОКОЕ слово хранения МК против 2-битного тернара AltAI.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import IN_DIM, OUT_DIM, RANDOM_SEED


@dataclass
class ProblemInstance:
    """Одна задача: тернарная весовая матрица W и бинарный вход x."""
    W: np.ndarray          # форма (OUT_DIM, IN_DIM), значения в {-1, 0, 1}, float64
    x: np.ndarray          # форма (IN_DIM,), значения в {0, 1}, int

    def copy(self) -> 'ProblemInstance':
        return ProblemInstance(W=self.W.copy(), x=self.x.copy())


def make_problem(seed: int = RANDOM_SEED,
                 in_dim: int = IN_DIM,
                 out_dim: int = OUT_DIM) -> ProblemInstance:
    """
    Сгенерировать воспроизводимую задачу.

    W — тернарная матрица; распределение сдвинуто так, чтобы по каждому выходу
    встречались и +1, и -1, и 0 (есть что «ронять» битфлипом), а argmax(W·x)
    был определён однозначно (нет ничьих) — иначе смена класса плохо измерима.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(10_000):
        W = rng.integers(-1, 2, size=(out_dim, in_dim)).astype(np.float64)
        x = rng.integers(0, 2, size=(in_dim,)).astype(np.int64)
        logits = W @ x
        order = np.sort(logits)[::-1]
        # требуем строгого победителя (нет ничьей за первое место)
        if order[0] - order[1] >= 1.0:
            return ProblemInstance(W=W, x=x)
    raise RuntimeError("Не удалось сгенерировать задачу со строгим победителем")


# ============================================================
# Эталонные алгоритмы (чистые, без сбоев)
# ============================================================
def linear_transform(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = W · x. Возвращает вектор логитов (float64)."""
    return np.asarray(W, dtype=np.float64) @ np.asarray(x, dtype=np.float64)


def threshold_classifier(W: np.ndarray, x: np.ndarray) -> int:
    """class = argmax(W · x). Возвращает индекс класса."""
    return int(np.argmax(linear_transform(W, x)))


# Реестр алгоритмов: имя -> (callable, тип выхода)
ALGORITHMS = {
    'linear_transform': {
        'fn': linear_transform,
        'output': 'vector',   # сравниваем по норме отклонения
    },
    'threshold_classifier': {
        'fn': threshold_classifier,
        'output': 'class',    # сравниваем по смене класса
    },
}
