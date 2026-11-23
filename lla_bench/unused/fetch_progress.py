"""Live preparation logs, including a heartbeat while third-party calls block."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path


class ProgressLines(list):
    """Keep the JSON summary's lines and flush each status to stdout and disk immediately."""

    def __init__(self, summary, interval=20):
        super().__init__()
        self.summary = Path(summary)
        self.summary.parent.mkdir(parents=True, exist_ok=True)
        self.summary.write_text("", encoding="utf-8")
        self.interval = interval
        self._lock = threading.RLock()
        self._stages = []
        self._last = "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _emit(self, message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.summary.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def append(self, message):
        with self._lock:
            super().append(str(message))
            self._last = str(message)
            self._emit(message)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._stages:
                    label, started = self._stages[-1]
                    self._emit(f"WAIT {label}: {time.monotonic() - started:.0f}s elapsed; "
                               f"last status: {self._last[:300]}")

    @contextmanager
    def stage(self, label):
        started = time.monotonic()
        with self._lock:
            self._stages.append((label, started))
            self.append(f"START {label}")
        try:
            yield
        except BaseException as error:
            self.append(f"FAILED {label} ({time.monotonic() - started:.1f}s): "
                        f"{type(error).__name__}: {error}")
            raise
        else:
            self.append(f"DONE {label} ({time.monotonic() - started:.1f}s)")
        finally:
            with self._lock:
                self._stages.pop()

    def close(self):
        self._stop.set()
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@contextmanager
def stage(lines, label):
    """Also accept the plain lists used by programmatic callers."""
    if isinstance(lines, ProgressLines):
        with lines.stage(label):
            yield
    else:
        yield


def run_stage(lines, label, function, *args, **kwargs):
    with stage(lines, label):
        return function(*args, **kwargs)
