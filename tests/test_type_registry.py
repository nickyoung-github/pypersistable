from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from pypersistable import RegistrationError, TypeRegistry, UnregisteredTypeError

# TypeRegistry is a singleton, so each test registers its own types and names


def test_register_registered_name_raises():
    class Old: ...

    class New: ...

    registry = TypeRegistry()
    registry.register("TestRegistered", Old)
    with pytest.raises(RegistrationError):
        registry.register("TestRegistered", New)


def test_register_replace():
    class Old: ...

    class New: ...

    registry = TypeRegistry()
    registry.register("TestReplaced", Old, root=Path("old"))
    registry.register("TestReplaced", New, root=Path("new"), replace=True)

    assert registry.get_type("TestReplaced") is New
    assert registry.registered_name(New) == "TestReplaced"
    assert registry.root(New) == Path("new")
    assert registry.types_for_root(Path("new")) == (New,)
    assert registry.types_for_root(Path("old")) == ()
    with pytest.raises(UnregisteredTypeError):
        registry.registered_name(Old)


class EntryPointTarget: ...


def test_register_entry_point():
    registry = TypeRegistry()
    entry_point = EntryPoint("target", f"{__name__}:EntryPointTarget", "pypersistable")
    registry.register("TestEntryPoint", entry_point)
    assert registry.get_type("TestEntryPoint") is EntryPointTarget
    assert registry.registered_name(EntryPointTarget) == "TestEntryPoint"
    # Registered already, by its entry point
    with pytest.raises(RegistrationError):
        registry.register("TestEntryPointAgain", EntryPointTarget)

    class Replacement: ...

    registry.register("TestEntryPoint", Replacement, replace=True)
    assert registry.get_type("TestEntryPoint") is Replacement
    with pytest.raises(UnregisteredTypeError):
        registry.registered_name(EntryPointTarget)


def test_unknown_root_raises():
    with pytest.raises(UnregisteredTypeError):
        TypeRegistry().types_for_root(Path("test-unknown-root"))
