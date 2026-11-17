"""Plug-ins: the connectors Vardex finds by name among the installed packages.

A model provider, a warehouse, a judge or a trainer is a plug-in: an object that a package
names in its entry points, under a group such as `vardex.providers`. Vardex looks it up by
name when a setting asks for it, and imports it only then. A provider nobody asked for is
never imported, and its SDK need not even be installed; adding one is installing a package
that declares it, with no change to the engine. Vardex's own connectors are found the same
way, through the entry points in its own pyproject.toml.
"""

from importlib.metadata import EntryPoint, entry_points
from typing import Any

from vardex.config import DEFAULT_PROVIDER

# Each kind of plug-in, its entry-point group, and what one is called in a message
GROUPS = {
    "providers": ("vardex.providers", "model provider"),
    "warehouses": ("vardex.warehouses", "warehouse"),
    "judges": ("vardex.judges", "judge"),
    "trainers": ("vardex.trainers", "trainer"),
}


class PluginError(Exception):
    """Raised when no installed package offers a plug-in by that name, or it cannot load."""


def available(kind: str) -> dict[str, EntryPoint]:
    """Every plug-in of a kind that the installed packages declare, by name."""
    group, _ = GROUPS[kind]
    return {point.name: point for point in entry_points(group=group)}


def load(kind: str, name: str) -> Any:
    """The plug-in of a kind with this name: imported now, and only now."""
    _, called = GROUPS[kind]
    found = available(kind)
    if name not in found:
        known = ", ".join(sorted(found)) or "none"
        raise PluginError(f"No {called} called {name!r} is installed. Installed: {known}.")
    try:
        return found[name].load()
    except ImportError as err:
        # The plug-in is declared, but a package it needs is not there: an extra that was not
        # installed, such as the SDK of a provider
        raise PluginError(
            f"The {called} {name!r} needs {err.name or 'a package'}, which is not installed. "
            f"For one that comes with Vardex, install its extra: pip install 'vardex[{name}]'."
        ) from err


def split_model(reference: str) -> tuple[str, str]:
    """A model reference as its provider and the model's own name: ollama/qwen3:8b is
    (ollama, qwen3:8b). Everything after the first slash is the name, so a server's model
    called Qwen/Qwen3-8B is openai/Qwen/Qwen3-8B. A name alone is the default provider's."""
    provider, slash, model = reference.partition("/")
    if not slash:
        return DEFAULT_PROVIDER, reference
    if not provider or not model:
        raise PluginError(f"{reference!r} is not a model: write it as provider/model")
    return provider, model
