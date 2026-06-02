"""
Воркер: собирает конфиг чипа AltAI под заданные веса, исполняет один вход на
golden model и печатает результат (active set + бинарный знаковый вектор) в виде
JSON в stdout.

Запускается в ОТДЕЛЬНОМ процессе на каждую инъекцию: golden model AltAI держит
глобальное состояние сетки ядер, которое не сбрасывается между повторными
build() в одном процессе (после ~2 сборок устройство перестаёт спайковать).
Свежий процесс = чистое устройство = корректный результат каждый раз.

Использование:
    python -m platforms.altai_worker <weights.npy> <x.npy> <n_ticks>
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np


def main():
    weights_path, x_path, n_ticks = sys.argv[1], sys.argv[2], int(sys.argv[3])
    W = np.load(weights_path).astype(np.int64)
    x = np.load(x_path).astype(np.int32)
    out_dim = W.shape[0]

    from platforms.altai_chip import build_chip_config, _silenced, _weights_hash
    from knp_ann2snn import Altai

    fault_dir = os.path.join(os.path.dirname(weights_path) or '.', '_build')
    os.makedirs(fault_dir, exist_ok=True)
    h = _weights_hash(W)
    mp = os.path.join(fault_dir, f'{h}.h5')
    cp = os.path.join(fault_dir, f'{h}.json')

    try:
        if not build_chip_config(W, mp, cp):
            print(json.dumps({'error': 'build failed'})); return
        altai = Altai()
        with _silenced():
            altai.build(config_path=cp, inference_type='gm')
            altai.prepare_spikes(x)
            altai.start_ticks(ticks=n_ticks)
            spikes = np.array(altai.get_spikes())
            altai.clear_input()
        flat = spikes.flatten()
        binary = [1.0 if np.any(flat == j) else 0.0 for j in range(out_dim)]
        active = [j for j in range(out_dim) if binary[j] > 0]
        print(json.dumps({'binary': binary, 'active': active}))
    finally:
        for f in (cp, mp):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass


if __name__ == '__main__':
    main()
