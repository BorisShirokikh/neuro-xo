from pathlib import Path
import time

from dpipe.torch import load_model_state

from ttt_lib.utils import choose_model


def wait_and_load_model_state(module, path, sleep_interval=5, max_wait_ticks=1000):
    path = Path(path)

    tick = 0
    while choose_model(path) is None:
        time.sleep(sleep_interval)
        tick += 1
        if tick >= max_wait_ticks:
            break

    load_model_state(module=module, path=path / choose_model(path))
