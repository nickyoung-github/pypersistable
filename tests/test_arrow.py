from dataclasses import dataclass
from decimal import Decimal
from ipaddress import IPv4Address
from typing import Annotated, Any, Literal, NamedTuple

import pyarrow as pa
import pytest
from pydantic import BaseModel

from pypersistable import ArrowConverter, UnsupportedTypeError


@dataclass
class Inner:
    value: Decimal
    tags: list[str]


class Model(BaseModel):
    inner: Inner
    counts: dict[str, int] | None = None


class Point(NamedTuple):  # Stands in for e.g. an attrs class or msgspec Struct
    x: float
    y: float


@dataclass
class Shape:
    centre: Point


def namedtuple_fields(typ: type):
    return (
        typ.__annotations__
        if issubclass(typ, tuple) and hasattr(typ, "_fields")
        else None
    )


def test_schema():
    assert ArrowConverter().arrow_schema(Model) == pa.schema(
        [
            (
                "inner",
                pa.struct(
                    [("value", pa.decimal128(38, 18)), ("tags", pa.list_(pa.string()))]
                ),
            ),
            ("counts", pa.map_(pa.string(), pa.int64())),
        ]
    )


def test_add_type():
    converter = ArrowConverter()
    converter.add_type(IPv4Address, pa.string(), str)
    assert converter.arrow_type(list[IPv4Address]) == pa.list_(pa.string())

    rows = converter.arrow_value([{"a": [IPv4Address("10.0.0.1")]}])
    assert pa.Table.from_pylist(rows).to_pylist() == [{"a": ["10.0.0.1"]}]


def test_add_type_overrides_builtin():
    converter = ArrowConverter()
    converter.add_type(int, pa.int32())
    assert converter.arrow_type(int) == pa.int32()
    assert (
        converter.arrow_type(bool) == pa.int32()
    )  # A subclass of int, added more recently


def test_add_struct():
    converter = ArrowConverter()
    with pytest.raises(UnsupportedTypeError):
        converter.arrow_schema(Shape)

    converter.add_struct(namedtuple_fields)
    assert converter.arrow_schema(Shape) == pa.schema(
        [("centre", pa.struct([("x", pa.float64()), ("y", pa.float64())]))]
    )


def test_add_handler():
    class AsText:
        pass

    def as_text(_: ArrowConverter, __: Any, metadata: tuple[Any, ...]):
        return pa.string() if any(isinstance(m, AsText) for m in metadata) else None

    @dataclass
    class Holder:
        value: Annotated[int, AsText()]

    converter = ArrowConverter()
    assert converter.arrow_schema(Holder).field("value").type == pa.int64()

    converter.add_handler(as_text)  # Also invalidates the cached schema
    assert converter.arrow_schema(Holder).field("value").type == pa.string()


@pytest.mark.parametrize(
    "hint",
    [int | str, Literal["a", 1], tuple[int, str], dict, Any, Point],
    ids=str,
)
def test_unsupported_types_raise(hint):
    with pytest.raises(UnsupportedTypeError):
        ArrowConverter().arrow_type(hint)
