from contextlib import contextmanager
import time
import numpy as np
from typing import Any, Generator, Optional


@contextmanager
def measure_time(desc: str) -> Generator[None, Any, None]:
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{desc} executed in: {elapsed:.2f} secs")


@contextmanager
def fixed_random_seed(seed: Optional[int] = None) -> Generator[None, Any, None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
