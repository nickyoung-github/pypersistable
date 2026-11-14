import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from dataclasses import InitVar, fields, make_dataclass
from functools import cache
from hashlib import sha512
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from orjson import OPT_SORT_KEYS, dumps
from pydantic import (
    ConfigDict,
    GetCoreSchemaHandler,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
)
from pydantic_core import CoreSchema, core_schema

from .object_store import ObjectStore, ReadBatch, UnknownObjectError
from .persistable_id import PersistableId
from .typed_cache import typed_cache


class _IdFieldIndicator: ...


T = TypeVar("T")
IdField = Annotated[T, _IdFieldIndicator]

_ID_ONLY = "id_only"


class Persistable(ABC):
    def __hash__(self):
        return hash(self.object_id)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def serialise(
            value: Persistable, serialiser: SerializerFunctionWrapHandler, info: SerializationInfo
        ) -> Any:
            # With id_only in the context, nested Persistables serialise as their object_id. Its value is
            # the root object being serialised, which is exempt: a Unique is its own id_type
            root = (info.context or {}).get(_ID_ONLY)
            if root is not None and value is not root:
                return str(value.object_id)
            return serialiser(value)

        return {
            **handler(source),
            "serialization": core_schema.wrap_serializer_function_ser_schema(serialise, info_arg=True),
        }

    @classmethod
    @abstractmethod
    def id_type(cls) -> type:
        """The type identifying objects of this class: a generated dataclass of the id fields
        for an `Identifiable`, the class itself for a `Unique`."""

    @classmethod
    @cache
    def __id_adapter(cls) -> TypeAdapter:
        return TypeAdapter(cls.id_type())

    @classmethod
    def _object_id(cls, source: Any) -> PersistableId:
        # source is either the id fields, as a dict, or an instance of id_type
        adapter = cls.__id_adapter()
        id_value = adapter.validate_python(source)
        values = adapter.dump_python(id_value, mode="json", context={_ID_ONLY: id_value})
        name = sha512(dumps(values, option=OPT_SORT_KEYS)).hexdigest()

        return PersistableId(name, typ=cls)

    @property
    def object_id(self) -> PersistableId:
        id_type = self.id_type()
        return self._object_id(
            self
            if isinstance(self, id_type)
            else {f.name: getattr(self, f.name) for f in fields(id_type)}
        )

    @overload
    @classmethod
    def read(
        cls,
        db: ObjectStore,
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
        db: ObjectStore,
        *,
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
        is_async: Literal[True],
        **kwargs,
    ) -> Coroutine[Any, Any, Self]: ...

    @classmethod
    def read(
        cls,
        db: ObjectStore,
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

        try:
            return db.read([object_id], effective_time, entry_time)[object_id]
        except KeyError:
            raise UnknownObjectError(cls, kwargs, object_id) from None

    def write(self, db: ObjectStore):
        db.write([self])


class Identifiable(Persistable):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.id_type()  # Validates the id fields

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
        if init_vars := [n for n, h in get_type_hints(cls).items() if isinstance(h, InitVar)]:
            raise TypeError(f"{cls!r}: a Unique cannot have InitVars: {', '.join(init_vars)}")

    @classmethod
    def id_type(cls) -> type[Self]:
        return cls
