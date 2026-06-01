from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed_stage(label: str, timings: dict[str, float]) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[label] = elapsed
        print(f"{label}: {elapsed:.1f}s")


def now_seconds() -> float:
    return time.perf_counter()
