"""Lazy lookup of adapter classes from ``"module:Class"`` registry entries.

The registries in ``callcheck.adapters.targets`` and ``callcheck.adapters.callers``
map a short name to the dotted path of a class. Importing the class only when it is
asked for keeps provider SDKs out of the import path of runs that do not use them.
"""

import importlib
from typing import Any


def build(registry: dict[str, str], kind: str, name: str, **kwargs: Any) -> Any:
    """Import the class registered under ``name`` and instantiate it with ``kwargs``.

    Raises ``KeyError`` naming the known entries when ``name`` is not registered.
    """
    if name not in registry:
        raise KeyError(f"unknown {kind} {name!r}; known: {sorted(registry)}")
    module_name, class_name = registry[name].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**kwargs)
