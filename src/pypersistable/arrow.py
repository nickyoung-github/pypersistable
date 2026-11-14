"""Arrow types for python type hints, e.g. for writing Parquet. Standalone: depends only on pyarrow and pydantic.

ArrowConverter handles python's scalar, container, Enum, Literal and Optional types, dataclasses and pydantic models.
Extend it with:
- add_type: a type (and its subclasses) as a given arrow type, with a conversion for values pyarrow does not accept
- add_struct: a function returning the fields of struct-like types, e.g. attrs classes or msgspec Structs
- add_handler: a function from any type hint and its Annotated metadata to an arrow type

Those added most recently take precedence, including over the built-in handling.
"""

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from types import NoneType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints
from uuid import UUID

import pyarrow as pa
from pydantic import AwareDatetime, BaseModel, NaiveDatetime

# (converter, hint, Annotated metadata) -> arrow type, or None if not handled
type TypeHandler = Callable[[ArrowConverter, Any, tuple[Any, ...]], pa.DataType | None]

# type -> {field name: type hint}, or None if not handled
type StructFields = Callable[[type], Mapping[str, Any] | None]

# When not constrained by max_digits/decimal_places
_DECIMAL_PRECISION, _DECIMAL_SCALE = 38, 18


class UnsupportedTypeError(TypeError):
    """A type with no arrow equivalent"""


def dataclass_fields(typ: type) -> Mapping[str, Any] | None:
    if not is_dataclass(typ):
        return None

    hints = get_type_hints(typ, include_extras=True)  # Keeping Annotated metadata
    return {f.name: hints[f.name] for f in fields(typ)}


def pydantic_fields(typ: type) -> Mapping[str, Any] | None:
    if not issubclass(typ, BaseModel):
        return None

    # pydantic moves Annotated metadata to FieldInfo.metadata
    return {
        name: Annotated[f.annotation, *f.metadata] if f.metadata else f.annotation
        for name, f in typ.model_fields.items()
    }


class ArrowConverter:
    def __init__(self):
        self.__handlers: list[TypeHandler] = []
        self.__structs: list[StructFields] = [pydantic_fields, dataclass_fields]
        # Checked in order, so subclasses precede their bases. A plain datetime is stored as UTC, keeping the instant:
        # it must be aware (see arrow_value). Use pydantic's NaiveDatetime for no time zone
        self.__types: dict[type, tuple[pa.DataType, Callable[[Any], Any] | None]] = {  # pyright: ignore[reportAttributeAccessIssue]
            NaiveDatetime: (pa.timestamp("us"), None),
            AwareDatetime: (pa.timestamp("us", tz="UTC"), None),
            bool: (pa.bool_(), None),
            int: (pa.int64(), None),
            float: (pa.float64(), None),
            str: (pa.string(), None),
            bytes: (pa.binary(), None),
            dt.datetime: (pa.timestamp("us", tz="UTC"), None),
            dt.date: (pa.date32(), None),
            dt.time: (pa.time64("us"), None),
            dt.timedelta: (pa.duration("us"), None),
            # Not pa.uuid(): pyarrow cannot convert python values to extension types nested in a struct
            UUID: (pa.string(), str),
        }
        self.__schemas: dict[type, pa.Schema] = {}
        self.__to_arrow: dict[type, Callable[[Any], Any] | None] = {}

    def add_type(
        self,
        typ: type,
        arrow_type: pa.DataType,
        to_arrow: Callable[[Any], Any] | None = None,
    ):
        """Store typ, and its subclasses, as arrow_type. to_arrow converts values which pyarrow does not accept"""
        self.__types.pop(typ, None)  # So that typ moves to the front
        self.__types = {typ: (arrow_type, to_arrow), **self.__types}
        self.__changed()

    def add_struct(self, struct_fields: StructFields):
        """struct_fields returns the field type hints of the struct-like types it handles, else None"""
        self.__structs.insert(0, struct_fields)
        self.__changed()

    def add_handler(self, handler: TypeHandler):
        """handler returns the arrow type for the type hints, with their Annotated metadata, it handles, else None"""
        self.__handlers.insert(0, handler)
        self.__changed()

    def arrow_schema(self, typ: type) -> pa.Schema:
        """The schema of a struct-like type, e.g. a dataclass or pydantic model"""
        if (schema := self.__schemas.get(typ)) is None:
            if (hints := self.struct_fields(typ)) is None:
                raise UnsupportedTypeError(f"Not a struct-like type: {typ}")
            schema = self.__schemas[typ] = pa.schema(self.__arrow_fields(hints))
        return schema

    def arrow_type(self, hint: Any, metadata: tuple[Any, ...] = ()) -> pa.DataType:
        origin, args = get_origin(hint), get_args(hint)

        if origin is Annotated:
            return self.arrow_type(args[0], (*metadata, *args[1:]))

        for handler in self.__handlers:
            if (arrow_type := handler(self, hint, metadata)) is not None:
                return arrow_type

        if origin is Union or origin is UnionType:
            if len(members := [a for a in args if a is not NoneType]) == 1:
                return self.arrow_type(members[0], metadata)
            raise UnsupportedTypeError(
                f"Unsupported union {hint}: only T | None is supported"
            )

        if origin is Literal:
            return self.__value_type(args, hint)

        if origin in (list, tuple, set, frozenset, Sequence, AbstractSet):
            if len(item_types := {a for a in args if a is not Ellipsis}) != 1:
                raise UnsupportedTypeError(
                    f"Unsupported {hint}: items must be of one type"
                )
            return pa.list_(self.arrow_type(item_types.pop()))

        if origin in (dict, Mapping):
            return pa.map_(*(self.arrow_type(a) for a in args))

        if isinstance(hint, type):
            if issubclass(hint, Enum):
                return self.__value_type([m.value for m in hint], hint)
            if issubclass(hint, Decimal):
                return pa.decimal128(*_decimal_precision_scale(metadata))
            if (entry := self.__type_entry(hint)) is not None:
                return entry[0]
            if (hints := self.struct_fields(hint)) is not None:
                return pa.struct(self.__arrow_fields(hints))

        raise UnsupportedTypeError(f"Unsupported type {hint}")

    def arrow_value(self, value: Any, arrow_type: pa.DataType) -> Any:
        """Plain python data, e.g. pydantic's python mode serialisation, as pyarrow accepts it for arrow_type.

        Structs must already be dicts: they are not converted. Raises ValueError for a naive datetime with a time zone
        aware arrow_type, which pyarrow would otherwise silently treat as UTC.
        """
        if value is None:
            return None
        if pa.types.is_struct(arrow_type):
            return {
                f.name: self.arrow_value(value.get(f.name), f.type) for f in arrow_type
            }
        if pa.types.is_map(arrow_type):
            key_type, item_type = arrow_type.key_type, arrow_type.item_type
            return {
                self.arrow_value(k, key_type): self.arrow_value(v, item_type)
                for k, v in value.items()
            }
        if pa.types.is_list(arrow_type):
            return [self.arrow_value(v, arrow_type.value_type) for v in value]
        if isinstance(value, Enum):
            value = value.value
        if (
            pa.types.is_timestamp(arrow_type)
            and arrow_type.tz is not None
            and value.utcoffset() is None
        ):
            raise ValueError(
                f"Naive datetime {value} for a time zone aware timestamp: use NaiveDatetime for no time zone"
            )

        typ = type(value)
        if typ not in self.__to_arrow:
            entry = self.__type_entry(typ)
            self.__to_arrow[typ] = entry[1] if entry is not None else None

        to_arrow = self.__to_arrow[typ]
        return value if to_arrow is None else to_arrow(value)

    def struct_fields(self, typ: type) -> Mapping[str, Any] | None:
        """The field type hints of a struct-like type, else None"""
        return next((f for s in self.__structs if (f := s(typ)) is not None), None)

    def __arrow_fields(self, hints: Mapping[str, Any]) -> list[pa.Field]:
        return [pa.field(name, self.arrow_type(hint)) for name, hint in hints.items()]

    def __type_entry(
        self, typ: type
    ) -> tuple[pa.DataType, Callable[[Any], Any] | None] | None:
        return next((e for t, e in self.__types.items() if issubclass(typ, t)), None)

    def __value_type(self, values: Sequence[Any], hint: Any) -> pa.DataType:
        # Enums and Literals are stored as their values, which must be of one type
        if len(types := {type(v) for v in values}) != 1:
            raise UnsupportedTypeError(
                f"Unsupported {hint}: values must be of one type"
            )
        return self.arrow_type(types.pop())

    def __changed(self):
        self.__schemas.clear()
        self.__to_arrow.clear()


def _decimal_precision_scale(metadata: tuple[Any, ...]) -> tuple[int, int]:
    # pydantic's max_digits/decimal_places, directly in the metadata or within a FieldInfo
    for constraint in (c for m in metadata for c in (m, *getattr(m, "metadata", ()))):
        digits = getattr(constraint, "max_digits", None)
        places = getattr(constraint, "decimal_places", None)
        if digits is not None or places is not None:
            precision = digits or _DECIMAL_PRECISION
            scale = places if places is not None else min(_DECIMAL_SCALE, precision)
            return precision, scale

    return _DECIMAL_PRECISION, _DECIMAL_SCALE
