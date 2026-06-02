"""
Фон-Неймановская платформа — numpy-модель обычного микроконтроллера.

Веса и состояние хранятся в ШИРОКИХ словах (float32 или int8), скалярное
произведение W·x считается ЯВНЫМ циклом накопления — чтобы битфлип можно было
вбросить ровно туда, куда бьёт SEU в железе:
  * weight       — бит в слове веса в RAM;
  * accumulator  — бит в регистре-аккумуляторе во время накопления;
  * pc           — бит в счётчике цикла (поток управления): искажённый индекс
                   уводит чтение мимо данных → SDC либо вылет за границы → крах.

Статус исполнения:
  'ok'    — посчиталось (возможно молча неверно — это и есть SDC);
  'fatal' — переполнение float (inf/nan/огромное число) от бита экспоненты/знака;
  'crash' — отказ потока управления (индекс вне диапазона) — класс отказов,
            которого у нейроморфа нет.
"""

from __future__ import annotations

import numpy as np

from config import FATAL_ABS_THRESHOLD
from fault_injection import Fault, flip_bit_float32, flip_bit_int


def quantize_weights(W: np.ndarray, dtype: str) -> np.ndarray:
    """Уложить тернарные веса в формат хранения МК."""
    if dtype == 'float32':
        return W.astype(np.float32)
    if dtype == 'int8':
        return np.clip(np.round(W), -128, 127).astype(np.int8)
    raise ValueError(f"неизвестный dtype: {dtype}")


class VonNeumannMCU:
    """Модель МК с весами фиксированной разрядности."""

    def __init__(self, W: np.ndarray, dtype: str = 'float32'):
        self.dtype = dtype
        self.W = quantize_weights(W, dtype)
        self.out_dim, self.in_dim = self.W.shape

    # --------------------------------------------------------
    def _flip_weight_word(self, w_value, bit: int):
        if self.dtype == 'float32':
            return np.float32(flip_bit_float32(float(w_value), bit))
        return np.int8(flip_bit_int(int(w_value), bit, 8))

    def _flip_accumulator(self, acc, bit: int):
        if self.dtype == 'float32':
            return np.float32(flip_bit_float32(float(acc), bit))
        # int8-веса → аккумулятор int32
        return np.int32(flip_bit_int(int(acc), bit, 32))

    # --------------------------------------------------------
    def linear_transform(self, x: np.ndarray, fault: Fault | None = None):
        """
        y = W · x явным циклом накопления. Возвращает (y: np.ndarray, status: str).
        """
        x = x.astype(self.W.dtype if self.dtype == 'int8' else np.float32)
        W = self.W.copy()

        # --- инъекция в слово веса (до вычисления) ---
        if fault is not None and fault.target == 'weight':
            W[fault.row, fault.col] = self._flip_weight_word(W[fault.row, fault.col], fault.bit)

        y = np.zeros(self.out_dim, dtype=np.float64)
        status = 'ok'

        # переполнение/inf при инъекции в экспоненту — ожидаемое поведение SEU,
        # не зашумляем вывод предупреждениями numpy
        with np.errstate(over='ignore', invalid='ignore'):
          for j in range(self.out_dim):
            acc = np.float32(0.0) if self.dtype == 'float32' else np.int32(0)
            for k in range(self.in_dim):
                kk = k
                # --- инъекция в счётчик цикла (поток управления) ---
                if (fault is not None and fault.target == 'pc'
                        and fault.row == j and fault.step == k):
                    kk = flip_bit_int(k, fault.bit, max(1, int(np.ceil(np.log2(self.in_dim))) + 1))
                    if kk < 0 or kk >= self.in_dim:
                        # искажённый индекс вне диапазона → доступ к чужой памяти/
                        # вылет → зависание/перезагрузка прошивки
                        status = 'crash'
                        y[:] = np.nan
                        return y, status

                acc = acc + W[j, kk] * x[kk]

                # --- инъекция в аккумулятор (во время накопления) ---
                if (fault is not None and fault.target == 'accumulator'
                        and fault.row == j and fault.step == k):
                    acc = self._flip_accumulator(acc, fault.bit)

            y[j] = float(acc)

        # --- проверка на фатальность (переполнение float) ---
        if not np.all(np.isfinite(y)) or np.any(np.abs(y) > FATAL_ABS_THRESHOLD):
            status = 'fatal'

        return y, status

    def threshold_classifier(self, x: np.ndarray, fault: Fault | None = None):
        """class = argmax(W · x). Возвращает (cls: int|None, status: str)."""
        y, status = self.linear_transform(x, fault)
        if status == 'crash':
            return None, status
        if not np.all(np.isfinite(y)):
            return None, 'fatal'
        return int(np.argmax(y)), status
