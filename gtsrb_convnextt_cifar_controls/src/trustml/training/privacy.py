from __future__ import annotations


def has_opacus() -> bool:
    try:
        import opacus  # noqa: F401
    except ImportError:
        return False
    return True

