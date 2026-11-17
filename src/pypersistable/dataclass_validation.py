from contextvars import ContextVar
from dataclasses import MISSING, fields
from functools import cached_property, partial, wraps
from inspect import getattr_static, isdatadescriptor
from types import MemberDescriptorType
from typing import Any, get_type_hints

from pydantic import GetCoreSchemaHandler, TypeAdapter, ValidatorFunctionWrapHandler
from pydantic_core import CoreSchema, core_schema

from .typed_cache import type_adapter

_SCHEMA_HOOK = "__get_pydantic_core_schema__"

# Set while pydantic constructs an instance, setting fields it has validated already
_CONSTRUCTING: ContextVar[bool] = ContextVar("_CONSTRUCTING", default=False)


class _ValidatingFieldDescriptor:
    def __init__(self, owner: type, name: str):
        self._owner = owner
        self._name = name

    @cached_property
    def _adapter(self) -> TypeAdapter:
        # Resolved on first use, as the type may refer to classes defined after its owner
        return type_adapter(
            get_type_hints(self._owner, include_extras=True)[self._name]
        )

    def _validate(self, value: Any) -> Any:
        if _CONSTRUCTING.get():
            return value
        return self._adapter.validate_python(value)

    def _no_attribute(self) -> AttributeError:
        return AttributeError(
            f"{self._owner.__qualname__!r} object has no attribute {self._name!r}"
        )


class _DictFieldDescriptor(_ValidatingFieldDescriptor):
    """Reads of the field unset, or of the class's attribute, fall back to its default, as for a dataclass"""

    def __init__(self, owner: type, name: str, default: Any):
        super().__init__(owner, name)
        self.__default = default

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is not None and self._name in obj.__dict__:
            return obj.__dict__[self._name]
        if self.__default is MISSING:
            raise self._no_attribute()
        return self.__default

    def __set__(self, obj: Any, value: Any):
        obj.__dict__[self._name] = self._validate(value)

    def __delete__(self, obj: Any):
        try:
            del obj.__dict__[self._name]
        except KeyError:
            raise self._no_attribute() from None


class _SlotFieldDescriptor(_ValidatingFieldDescriptor):
    def __init__(self, owner: type, name: str, slot: MemberDescriptorType):
        super().__init__(owner, name)
        self.slot = slot

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        return self.slot.__get__(obj, owner)

    def __set__(self, obj: Any, value: Any):
        self.slot.__set__(obj, self._validate(value))

    def __delete__(self, obj: Any):
        self.slot.__delete__(obj)


def _constructing(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    token = _CONSTRUCTING.set(True)
    try:
        return handler(value)
    finally:
        _CONSTRUCTING.reset(token)


def _validating_post_init(post_init: Any) -> Any:
    """Validates assignments in post_init also when pydantic, constructing an instance, calls it"""

    @wraps(post_init)
    def __post_init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        token = _CONSTRUCTING.set(False)
        try:
            return post_init(self, *args, **kwargs)
        finally:
            _CONSTRUCTING.reset(token)

    __post_init__.validating = True  # pyright: ignore[reportAttributeAccessIssue]
    return __post_init__


class _SchemaHook:
    """
    A class's __get_pydantic_core_schema__: marks pydantic constructing its instances, wrapping the class's own hook,
    else that inherited, else pydantic's default
    """

    def __init__(self, owner: type, own: Any):
        self.__owner = owner
        self.__own = own

    def __get__(self, obj: Any, cls: type) -> Any:
        return partial(self.__schema, cls)

    def __schema(
        self, cls: type, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        hook = (
            self.__own.__get__(None, cls)
            if self.__own is not None
            else getattr(super(self.__owner, cls), _SCHEMA_HOOK, None)
        )
        schema = hook(source, handler) if hook is not None else handler(source)
        return core_schema.no_info_wrap_validator_function(_constructing, schema)


def _install_descriptors(cls: type):
    for f in fields(cls):
        # The attribute in effect for the field: a slot, else values are kept in the instance's __dict__
        attr = getattr_static(cls, f.name, None)
        if isinstance(attr, _SlotFieldDescriptor):
            attr = attr.slot

        if isinstance(attr, MemberDescriptorType):
            field = _SlotFieldDescriptor(cls, f.name, attr)
        elif isdatadescriptor(attr) and not isinstance(
            attr, _ValidatingFieldDescriptor
        ):
            # E.g. a property: it handles assignment itself
            continue
        else:
            field = _DictFieldDescriptor(cls, f.name, f.default)
        setattr(cls, f.name, field)

    setattr(cls, _SCHEMA_HOOK, _SchemaHook(cls, vars(cls).get(_SCHEMA_HOOK)))

    # Inherited, it may be wrapped already
    post_init = getattr(cls, "__post_init__", None)
    if post_init is not None and not getattr(post_init, "validating", False):
        cls.__post_init__ = _validating_post_init(post_init)


class ValidatingDataclassMeta(type):
    """Validates, in strict mode, each value assigned to a field of a dataclass of this metaclass"""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # The class made by dataclass(slots=True), its fields known
        if "__dataclass_fields__" in namespace:
            _install_descriptors(cls)
        return cls

    def __setattr__(cls, name: str, value: Any):
        super().__setattr__(name, value)
        # Set by dataclass once it knows the fields, before making __init__. With slots, it then replaces cls
        if (
            name == "__dataclass_fields__" and not cls.__dataclass_params__.slots  # pyright: ignore[reportAttributeAccessIssue]
        ):
            _install_descriptors(cls)
