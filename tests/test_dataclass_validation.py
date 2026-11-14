import dataclasses
from dataclasses import FrozenInstanceError, InitVar, dataclass, field
from typing import Annotated, ClassVar

import pytest
from annotated_types import Gt
from pydantic import AfterValidator, TypeAdapter, ValidationError, ValidationInfo
from pydantic_core import core_schema

from pypersistable.dataclass_validation import ValidatingDataclassMeta
from pypersistable.typed_cache import type_adapter


@dataclass
class Plain(metaclass=ValidatingDataclassMeta):
    i: int
    s: str = "a"
    ints: list[int] = field(default_factory=list)
    c: ClassVar[int] = 0
    v: InitVar[int] = 0


@dataclass(slots=True)
class Slotted(metaclass=ValidatingDataclassMeta):
    i: int
    s: str = "a"


@dataclass(slots=True)
class SlottedChild(Slotted):
    f: float = 0.0


@dataclass
class Narrowed(Plain):
    i: Annotated[int, Gt(0)] = 1


@dataclass(frozen=True)
class Frozen(metaclass=ValidatingDataclassMeta):
    i: int


@dataclass
class Node(metaclass=ValidatingDataclassMeta):
    name: str
    next: "Node | None" = None


@dataclass
class BadDefault(metaclass=ValidatingDataclassMeta):
    i: int = "a"  # pyright: ignore[reportAssignmentType]


class Doubling:
    def __set_name__(self, owner: type, name: str):
        self._name = f"_{name}"

    def __get__(self, obj: object, owner: type | None = None) -> int:
        return 1 if obj is None else getattr(obj, self._name)

    def __set__(self, obj: object, value: int):
        setattr(obj, self._name, value * 2)


@dataclass
class HasDescriptor(metaclass=ValidatingDataclassMeta):
    d: Doubling = Doubling()
    i: int = 0


@pytest.mark.parametrize("cls", (Plain, Slotted, SlottedChild))
def test_validates(cls):
    obj = cls(i=1)
    assert (obj.i, obj.s) == (1, "a")

    obj.s = "b"
    assert obj.s == "b"
    with pytest.raises(ValidationError):
        obj.s = 1
    # Strict: no coercion
    with pytest.raises(ValidationError):
        cls(i="1")


def test_slots_kept():
    assert not hasattr(SlottedChild(i=1), "__dict__")


def test_defaults_validated():
    with pytest.raises(ValidationError):
        BadDefault()


def test_class_and_init_vars_not_validated():
    obj = Plain(i=1, v="a")  # pyright: ignore[reportArgumentType]
    obj.c = "a"  # pyright: ignore[reportAttributeAccessIssue]


def test_validated_value_stored():
    ints = [1]
    # pydantic validates a list as a copy
    assert Plain(i=1, ints=ints).ints is not ints


def test_frozen():
    obj = Frozen(i=1)
    with pytest.raises(FrozenInstanceError):
        obj.i = 2  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError):
        Frozen(i="1")  # pyright: ignore[reportArgumentType]


def test_forward_reference():
    Node("a", Node("b"))
    with pytest.raises(ValidationError):
        Node("a", "b")  # pyright: ignore[reportArgumentType]


def test_subclass_field_type():
    Plain(i=0)
    with pytest.raises(ValidationError):
        Narrowed(i=0)


@pytest.mark.parametrize("slots", (False, True))
def test_init_false(slots):
    @dataclass(slots=slots)
    class Uninitialised(metaclass=ValidatingDataclassMeta):
        no_default: int = field(init=False)
        default: int = field(init=False, default=1)

    obj = Uninitialised()
    # As for a dataclass
    with pytest.raises(AttributeError):
        _ = obj.no_default
    assert obj.default == 1

    obj.no_default = obj.default = 2
    assert (obj.no_default, obj.default) == (2, 2)
    with pytest.raises(ValidationError):
        obj.default = "a"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("cls", (Plain, Slotted))
def test_delete(cls):
    obj = cls(i=1)
    del obj.i, obj.s
    # As for a dataclass: falls back to the default, if any
    with pytest.raises(AttributeError):
        _ = obj.i
    if cls is Plain:
        assert obj.s == "a"
    else:
        # A slot, with no class attribute to fall back to
        with pytest.raises(AttributeError):
            _ = obj.s
    with pytest.raises(AttributeError):
        del obj.i

    obj.i = 2
    assert obj.i == 2
    with pytest.raises(ValidationError):
        obj.i = "2"


def test_class_attributes_are_defaults():
    # As for a dataclass without slots
    assert Plain.s == "a"
    with pytest.raises(AttributeError):
        _ = Plain.i
    # Its default_factory's
    with pytest.raises(AttributeError):
        _ = Plain.ints


def test_delete_init_false():
    @dataclass
    class Uninitialised(metaclass=ValidatingDataclassMeta):
        default: int = field(init=False, default=1)

    obj = Uninitialised()
    obj.default = 2
    del obj.default
    # As for a dataclass, falls back to the default
    assert obj.default == 1


def test_validates_before_instantiation():
    @dataclass
    class Uninstantiated(metaclass=ValidatingDataclassMeta):
        i: int

    # As, e.g., unpickling makes an instance
    obj = object.__new__(Uninstantiated)
    with pytest.raises(ValidationError):
        obj.i = "1"  # pyright: ignore[reportAttributeAccessIssue]


def test_data_descriptor_fields_excluded():
    obj = HasDescriptor()
    assert obj.d == 2
    # Assigned by the descriptor, unvalidated
    assert HasDescriptor(d="a").d == "aa"  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError):
        HasDescriptor(i="a")  # pyright: ignore[reportArgumentType]


def _needs_context(value: int, info: ValidationInfo) -> int:
    if not info.context:
        raise ValueError("No context")
    return value


@dataclass(slots=True)
class NeedsContext(metaclass=ValidatingDataclassMeta):
    i: Annotated[int, AfterValidator(_needs_context)]


def test_not_revalidated_when_constructed_by_pydantic():
    # pydantic sets each slot, its value validated already, with the context
    obj = TypeAdapter(NeedsContext).validate_python({"i": 1}, context={"a": 1})
    assert obj.i == 1
    # Assignment is validated, without a context
    with pytest.raises(ValidationError, match="No context"):
        obj.i = 2


@pytest.mark.parametrize("slots", (False, True))
def test_post_init_validated(slots):
    @dataclass
    class Unvalidated:
        i: int

        def __post_init__(self):
            self.i = str(self.i)  # pyright: ignore[reportAttributeAccessIssue]

    @dataclass(slots=slots)
    class Own(metaclass=ValidatingDataclassMeta):
        i: int

        def __post_init__(self):
            self.i = str(self.i)  # pyright: ignore[reportAttributeAccessIssue]

    @dataclass(slots=slots)
    class Inherited(Unvalidated, metaclass=ValidatingDataclassMeta):
        pass

    # Also when pydantic constructs the instance, which calls __post_init__
    for cls in (Own, Inherited):
        with pytest.raises(ValidationError):
            cls(i=1)
        with pytest.raises(ValidationError):
            TypeAdapter(cls).validate_python({"i": 1})


def test_post_init_super_with_slots():
    @dataclass(slots=True)
    class Base(metaclass=ValidatingDataclassMeta):
        i: int

        def __post_init__(self):
            self.i += 1

    @dataclass(slots=True)
    class Child(Base):
        j: int = 0

        def __post_init__(self):
            super().__post_init__()
            self.j = self.i

    @dataclass(slots=True)
    class GrandChild(Child):
        pass

    assert Child(i=1).j == TypeAdapter(Child).validate_python({"i": 1}).j == 2
    # Inherited wrapped already, so not wrapped again
    assert "__post_init__" not in vars(GrandChild)


def test_own_schema_hook_kept():
    @dataclass(slots=True)
    class Doubled(metaclass=ValidatingDataclassMeta):
        i: int

        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            return core_schema.no_info_after_validator_function(
                lambda obj: dataclasses.replace(obj, i=obj.i * 2), handler(source)
            )

    assert TypeAdapter(Doubled).validate_python({"i": 1}).i == 2


def test_type_adapter_cached():
    assert type_adapter(list[int]) is type_adapter(list[int])
