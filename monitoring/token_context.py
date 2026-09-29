"""
monitoring/token_context.py

Thread-local token counter. Each request runs _run_graph() in a dedicated
thread (via run_in_executor), so threading.local() gives per-request isolation.
"""
import threading

_local = threading.local()


def reset() -> None:
    """Call at the START of every request thread to clear any leftover state."""
    _local.input_tokens = 0
    _local.output_tokens = 0


def add(input_tokens: int, output_tokens: int) -> None:
    if not hasattr(_local, "input_tokens"):
        reset()
    _local.input_tokens += input_tokens
    _local.output_tokens += output_tokens


def get() -> dict:
    return {
        "input": getattr(_local, "input_tokens", 0),
        "output": getattr(_local, "output_tokens", 0),
    }
