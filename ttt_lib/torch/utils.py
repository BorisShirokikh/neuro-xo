from pathlib import Path
import time
import os

from dpipe.torch import load_model_state


def wait_and_load_model_state(module, exp_path, ep, sleep_interval=5, max_wait_ticks=1000):
    same_model_path = Path(exp_path) / f'model_{ep}.pth'

    tick = 0
    while not os.path.exists(same_model_path):
        time.sleep(sleep_interval)
        tick += 1
        if tick >= max_wait_ticks:
            break

    load_model_state(module=module, path=same_model_path)
