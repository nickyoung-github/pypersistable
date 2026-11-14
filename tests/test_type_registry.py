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
