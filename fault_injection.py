"""
Инъекторы однобитового сбоя (модель SEU).

Содержит:
  * битовые примитивы — переворот одного бита в IEEE-754 float32, в целом
    числе заданной разрядности (two's complement) и в 2-битном тернарном весе;
  * dataclass `Fault` — описание одной инъекции (куда и в какой бит бить);
  * сэмплеры случайных мишеней для фон-Неймана и нейроморфа.

Битовые примитивы намеренно работают на уровне представления слова в памяти —
ровно так, как SEU переворачивает физический бит ячейки/регистра.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
import numpy as np


# ============================================================
# Битовые примитивы
# ============================================================
def flip_bit_float32(value: float, bit: int) -> float:
    """
    Перевернуть бит `bit` (0..31) в IEEE-754 float32.
      биты 0..22  — мантисса
      биты 23..30 — экспонента (бит 30 — старший: ±2^64)
      бит  31     — знак
    """
    if not 0 <= bit <= 31:
        raise ValueError("bit float32 должен быть в диапазоне 0..31")
    packed = struct.pack('<f', float(np.float32(value)))
    u = struct.unpack('<I', packed)[0]
    u ^= (1 << bit)
    return struct.unpack('<f', struct.pack('<I', u))[0]


def flip_bit_int(value: int, bit: int, width: int) -> int:
    """
    Перевернуть бит `bit` (0..width-1) в целом числе разрядности `width`
    в дополнительном коде (two's complement). Старший бит — знаковый.
    """
    if not 0 <= bit < width:
        raise ValueError(f"bit должен быть в диапазоне 0..{width - 1}")
    mask = (1 << width) - 1
    u = int(value) & mask
    u ^= (1 << bit)
    if u >= (1 << (width - 1)):     # знаковый бит установлен → отрицательное
        u -= (1 << width)
    return u


# Кодирование тернарного веса в 2 битах (two's complement, диапазон -2..1):
#   0  -> 0b00,  1 -> 0b01,  -1 -> 0b11,  (0b10 = -2 — невалидный паттерн)
def flip_bit_ternary(weight: int, bit: int) -> int:
    """
    Перевернуть один из 2 битов тернарного веса {-1,0,1}.

    Это ключевой механизм устойчивости нейроморфа: вес хранится в 2 битах,
    поэтому любой битфлип даёт ОГРАНИЧЕННОЕ изменение. Невалидный паттерн -2
    клиппируется к ближайшему валидному тернару (-1), как это делает железо.
    """
    if bit not in (0, 1):
        raise ValueError("у тернарного веса всего 2 бита (0 или 1)")
    u = int(weight) & 0b11
    u ^= (1 << bit)
    if u >= 2:          # знаковый бит установлен
        u -= 4          # -> -2 (для 0b10) или -1 (для 0b11)
    if u == -2:         # невалидный паттерн → клиппинг к ближайшему валидному
        u = -1
    return u


def flip_bit_membrane(value: float, bit: int, scale: float = 256.0,
                      width: int = 16) -> float:
    """
    Перевернуть бит в мембранном потенциале нейрона. Потенциал квантуется в
    знаковое слово int`width` (фиксированная точка) — так он и хранится в SRAM
    ядра. После переворота возвращаем обратно в float.
    """
    q = int(round(float(value) * scale))
    lim = (1 << (width - 1))
    q = max(-lim, min(lim - 1, q))      # насыщение в диапазон int16
    q = flip_bit_int(q, bit, width)
    return q / scale


# ============================================================
# Описание одной инъекции
# ============================================================
@dataclass
class Fault:
    """Описание одного битфлипа."""
    domain: str                 # 'vn' (фон-Нейман) | 'nm' (нейроморф)
    target: str                 # weight/accumulator/pc | synapse/membrane
    bit: int                    # индекс перевёрнутого бита
    row: int = -1               # индекс выходного нейрона / строки W
    col: int = -1               # индекс входа / столбца W
    step: int = -1              # шаг накопления (vn) или номер тика (nm)
    neuron: int = -1            # индекс нейрона (для membrane)
    meta: dict = field(default_factory=dict)

    def short(self) -> str:
        loc = []
        if self.row >= 0:
            loc.append(f"row={self.row}")
        if self.col >= 0:
            loc.append(f"col={self.col}")
        if self.step >= 0:
            loc.append(f"step={self.step}")
        if self.neuron >= 0:
            loc.append(f"n={self.neuron}")
        return f"{self.domain}:{self.target}[bit={self.bit}{(',' + ','.join(loc)) if loc else ''}]"


# ============================================================
# Сэмплеры случайных мишеней
# ============================================================
def sample_vn_fault(rng: np.random.Generator, target: str,
                    out_dim: int, in_dim: int, dtype: str) -> Fault:
    """Случайная мишень для фон-Неймановского МК."""
    word_bits = 32 if dtype == 'float32' else 8   # разрядность слова веса
    if target == 'weight':
        return Fault('vn', 'weight',
                     bit=int(rng.integers(0, word_bits)),
                     row=int(rng.integers(0, out_dim)),
                     col=int(rng.integers(0, in_dim)))
    if target == 'accumulator':
        # аккумулятор: float32 (32 бита) либо int32 при int8-весах
        acc_bits = 32
        return Fault('vn', 'accumulator',
                     bit=int(rng.integers(0, acc_bits)),
                     row=int(rng.integers(0, out_dim)),
                     step=int(rng.integers(0, in_dim)))
    if target == 'pc':
        # счётчик цикла: бьём в младшие биты индекса (0..log2(in_dim)+1)
        idx_bits = max(1, int(np.ceil(np.log2(in_dim))) + 1)
        return Fault('vn', 'pc',
                     bit=int(rng.integers(0, idx_bits)),
                     row=int(rng.integers(0, out_dim)),
                     step=int(rng.integers(0, in_dim)))
    raise ValueError(f"неизвестная цель vn: {target}")


def sample_nm_fault(rng: np.random.Generator, target: str,
                    out_dim: int, in_dim: int, n_ticks: int,
                    membrane_bits: int) -> Fault:
    """Случайная мишень для нейроморфа AltAI."""
    if target == 'synapse':
        return Fault('nm', 'synapse',
                     bit=int(rng.integers(0, 2)),     # тернарный вес = 2 бита
                     row=int(rng.integers(0, out_dim)),
                     col=int(rng.integers(0, in_dim)))
    if target == 'membrane':
        return Fault('nm', 'membrane',
                     bit=int(rng.integers(0, membrane_bits)),
                     neuron=int(rng.integers(0, out_dim)),
                     step=int(rng.integers(0, n_ticks)))
    raise ValueError(f"неизвестная цель nm: {target}")
