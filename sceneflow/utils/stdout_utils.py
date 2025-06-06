import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr (e.g., from noisy libraries like MMOCR)."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
