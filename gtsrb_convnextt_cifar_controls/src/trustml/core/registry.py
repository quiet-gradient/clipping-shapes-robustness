from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, key: str, builder: Callable[..., T]) -> None:
        normalized = key.lower()
        if normalized in self._items:
            raise ValueError(f"{self.name} registry already has key: {key}")
        self._items[normalized] = builder

    def build(self, key: str, *args, **kwargs) -> T:
        normalized = key.lower()
        if normalized not in self._items:
            known = ", ".join(sorted(self._items))
            raise ValueError(f"Unknown {self.name}: {key}. Known: {known}")
        return self._items[normalized](*args, **kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)

