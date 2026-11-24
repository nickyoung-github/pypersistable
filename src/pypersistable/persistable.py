import datetime as dt
from collections.abc import Coroutine
from contextvars import ContextVar
from dataclasses import MISSING, InitVar, fields, is_dataclass, make_dataclass
from functools import cache
from hashlib import sha512
from importlib.util import find_spec
from inspect import getattr_static, isdatadescriptor
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)
from weakref import WeakSet

from orjson import OPT_SORT_KEYS, dumps
from pydantic import (
    ConfigDict,
    GetCoreSchemaHandler,
    SerializationInfo,
    SerializerFunctionWrapHandler,
)
from pydantic_core import CoreSchema, PydanticSerializationError, core_schema

from .deferred_post_init import defer_post_init
from .persistable_id import PersistableId
from .persistable_store import PersistableStore, ReadBatch, UnknownObjectError
from .typed_cache import type_adapter, typed_cache


class _IdFieldIndicator: ...


T = TypeVar("T")
IdField = Annotated[T, _IdFieldIndicator]

_ID_ONLY = "id_only"


# The objects whose ids are being computed, outermost first
_COMPUTING_IDS: ContextVar[tuple[Any, ...]] = ContextVar("_COMPUTING_IDS", default=())


# The registered classes: their fields are, as attributes of the class, their query columns (see query)
_QUERYABLE: WeakSet[type] = WeakSet()


class _QueryField:
    """Replaces a registered class's attribute for a field, behaving as it, but for the class itself, for which it is
    the field's query column. Not for its subclasses until they are registered: e.g. dataclass takes the attribute of
    a class it is making to be the field's default"""

    def __init__(self, name: str, wrapped: Any):
        self.name = name
        self.__wrapped__ = wrapped

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None and owner is not None and owner in _QUERYABLE:
            from .query import _attribute  # Lazily: it needs sqlalchemy

            return _attribute(owner, owner, (self.name,))

        wrapped = self.__wrapped__
        if hasattr(type(wrapped), "__get__"):
            return wrapped.__get__(obj, owner)
        if wrapped is MISSING:
            raise AttributeError(self.name)
        return wrapped


class _QueryDataField(_QueryField):
    """Of a data descriptor, e.g. a slot: unlike others, it takes precedence over the values of instances"""

    def __set__(self, obj: Any, value: Any):
        self.__wrapped__.__set__(obj, value)

    def __delete__(self, obj: Any):
        self.__wrapped__.__delete__(obj)


@cache
def _has_sqlalchemy() -> bool:
    return find_spec("sqlalchemy") is not None


class Persistable:
    __object_entry_time = None  # Not annotated, so that it is not a field

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Called by dataclasses only: harmless otherwise
        defer_post_init(cls, "__post_init__")

    @classmethod
    def _registered(cls):
        """Called by TypeRegistry, so once cls is made, e.g. by dataclass: makes its fields its query columns"""
        if cls in _QUERYABLE or not _has_sqlalchemy():
            return
        names = (
            [f.name for f in fields(cls)]
            if is_dataclass(cls)
            else getattr(cls, "model_fields", ())
        )
        for name in names:
            attr = getattr_static(cls, name, MISSING)
            # Else inherited from a registered base
            if not isinstance(attr, _QueryField):
                field = _QueryDataField if isdatadescriptor(attr) else _QueryField
                setattr(cls, name, field(name, attr))
        _QUERYABLE.add(cls)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        # For BaseModels, once pydantic has made the class: it wraps model_post_init itself if there are private attributes
        super().__pydantic_init_subclass__(**kwargs)  # pyright: ignore[reportAttributeAccessIssue]
        if getattr(cls, "__pydantic_post_init__", None):
            defer_post_init(cls, "model_post_init")

    def __hash__(self):
        return hash(self.object_id)

    @property
    def object_entry_time(self) -> dt.datetime | None:
        """The entry time of the persisted version of this object, if it was read or written, else None"""
        return self.__object_entry_time

    def _set_object_entry_time(self, entry_time: dt.datetime):
        # Not setattr, which frozen dataclasses disallow
        object.__setattr__(self, "_Persistable__object_entry_time", entry_time)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def serialise(
            value: Persistable,
            serialiser: SerializerFunctionWrapHandler,
            info: SerializationInfo,
        ) -> Any:
            # With id_only in the context, nested Persistables serialise as their object_id. Its value is
            # the root object being serialised, which is exempt: a Unique is its own id_type
            root = (info.context or {}).get(_ID_ONLY)
            if root is not None and value is not root:
                return str(value.object_id)
            return serialiser(value)

        # Serialising via its schema, keeping any serialiser of its own, e.g. a model_serializer
        schema = handler(source)
        return {
            **schema,
            "serialization": core_schema.wrap_serializer_function_ser_schema(
                serialise, info_arg=True, schema=schema
            ),
        }  # pyright: ignore[reportReturnType]

    @classmethod
    def id_type(cls) -> type:
        """The type identifying objects of this class: a generated dataclass of the id fields
        for an `Identifiable`, the class itself for a `Unique`."""
        raise NotImplementedError

    @classmethod
    def _object_id(cls, source: Any) -> PersistableId:
        # source is either the id fields, as a dict, or an instance of id_type
        adapter = type_adapter(cls.id_type())
        id_value = adapter.validate_python(source)
        values = adapter.dump_python(
            id_value, mode="json", context={_ID_ONLY: id_value}
        )
        name = sha512(dumps(values, option=OPT_SORT_KEYS)).hexdigest()

        return PersistableId(name, typ=cls)

    @property
    def object_id(self) -> PersistableId:
        computing = _COMPUTING_IDS.get()
        if (
            start := next((i for i, o in enumerate(computing) if o is self), None)
        ) is not None:
            cycle = " -> ".join(
                type(o).__qualname__ for o in (*computing[start:], self)
            )
            # Raised as is through the serialisers of the objects referencing self, unlike other exceptions
            raise PydanticSerializationError(f"Reference cycle in id fields: {cycle}")

        token = _COMPUTING_IDS.set((*computing, self))
        try:
            id_type = self.id_type()
            return self._object_id(
                self
                if isinstance(self, id_type)
                else {f.name: getattr(self, f.name) for f in fields(id_type)}
            )
        finally:
            _COMPUTING_IDS.reset(token)

    @overload
    @classmethod
    def read(
        cls,
        db: PersistableStore,
        /,
        *,
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
        is_async: Literal[False] = False,
        **kwargs,
    ) -> Self: ...

    @overload
    @classmethod
    def read(
        cls,
        db: PersistableStore,
        /,
        *,
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
        is_async: Literal[True],
        **kwargs,
    ) -> Coroutine[Any, Any, Self]: ...

    @classmethod
    def read(
        cls,
        db: PersistableStore,
        /,
        *,
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
        is_async: bool = False,
        **kwargs,
    ) -> Self | Coroutine[Any, Any, Self]:
        object_id = cls._object_id(kwargs)
        if is_async:
            return ReadBatch.current().enqueue(
                db, cls, object_id, kwargs, effective_time, entry_time
            )

        # Not within the try: a KeyError raised reading is not an unknown object
        objects = db.read([object_id], effective_time, entry_time)
        try:
            # Of cls, as its id says
            return cast(Self, objects[object_id])
        except KeyError:
            raise UnknownObjectError(cls, kwargs, object_id) from None

    def write(
        self,
        db: PersistableStore,
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
        by_reference: bool = True,
    ) -> None:
        db.write(
            [self],
            effective_time=effective_time,
            force_overwrite=force_overwrite,
            by_reference=by_reference,
        )

    @classmethod
    def delete(
        cls,
        db: PersistableStore,
        /,
        *,
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
        **kwargs,
    ) -> None:
        """Reads the object with the given id fields, then deletes it: see del_"""
        cls.read(db, **kwargs).del_(db, effective_time, force_overwrite)

    def del_(
        self,
        db: PersistableStore,
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
    ) -> None:
        """Marks this object deleted: see PersistableStore.delete. Writing it again restores it"""
        db.delete(
            [self], effective_time=effective_time, force_overwrite=force_overwrite
        )


def _check_id_type(cls: type[Persistable]):
    try:
        cls.id_type()  # Validates the id fields
    except NameError:
        pass  # A forward reference, e.g. to cls itself, which is not defined yet: validated on first use instead


class Identifiable(Persistable):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _check_id_type(cls)

    @classmethod
    @typed_cache
    def id_type(cls) -> type:
        id_hints = {}
        for name, hint in get_type_hints(cls, include_extras=True).items():
            # Unwrap InitVar[IdField[X]], so that it is caught along with IdField[InitVar[X]]
            inner = hint.type if (init_var := isinstance(hint, InitVar)) else hint
            if get_origin(inner) is Annotated and _IdFieldIndicator in get_args(inner):
                if init_var or isinstance(get_args(inner)[0], InitVar):
                    raise TypeError(f"{cls!r}: IdField {name} cannot be an InitVar")
                id_hints[name] = hint

        if not id_hints:
            raise TypeError(f"{cls!r} does not define any id fields")

        return make_dataclass(
            f"{cls.__name__}Id",
            id_hints.items(),
            frozen=True,
            # Unknown fields must be an error, rather than being silently dropped by validation
            namespace={"__pydantic_config__": ConfigDict(extra="forbid")},
        )


class Unique(Persistable):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _check_id_type(cls)

    @classmethod
    @typed_cache
    def id_type(cls) -> type[Self]:
        if init_vars := [
            n for n, h in get_type_hints(cls).items() if isinstance(h, InitVar)
        ]:
            raise TypeError(
                f"{cls!r}: a Unique cannot have InitVars: {', '.join(init_vars)}"
            )
        return cls
