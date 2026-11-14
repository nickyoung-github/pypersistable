import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

import pytest
from models import Cat, Counterparty, CustomerBook, Dog, Kennel, registry
from pyarrow import ArrowInvalid, ArrowNotImplementedError
from pydantic_core import PydanticSerializationError

from pypersistable import FSStore, Identifiable, IdField, RecordFormat
from pypersistable.object_store import UnknownObjectError


@pytest.fixture(params=list(RecordFormat), ids=lambda f: f.name)
def store(request, tmp_path) -> FSStore:
    return FSStore(f"file://{tmp_path}", record_format=request.param)


@pytest.fixture
def json_store(tmp_path) -> FSStore:
    return FSStore(f"file://{tmp_path}", record_format=RecordFormat.Json)


def test_round_trip(store):
    counterparty = Counterparty(name="Acme", location="NW3")
    counterparty.write(store)
    assert Counterparty.read(store, name="Acme") == counterparty


def test_round_trip_with_nested_persistable(store):
    book = CustomerBook(
        counterparty=Counterparty(name="Acme"), desk="D", coverage=("a", "b")
    )
    book.write(store)
    assert CustomerBook.read(store, counterparty=book.counterparty, desk="D") == book


def test_read_unknown_object_raises(store):
    with pytest.raises(UnknownObjectError):
        Counterparty.read(store, name="Nobody")  # Nothing of this type persisted

    Counterparty(name="Acme").write(store)
    with pytest.raises(UnknownObjectError):
        Counterparty.read(store, name="Nobody")


def test_latest_version_is_read(store):
    Counterparty(name="Acme", location="NW3").write(store)
    Counterparty(name="Acme", location="SW1").write(store)
    assert Counterparty.read(store, name="Acme").location == "SW1"


def test_read_as_of_entry_time(store):
    Counterparty(name="Acme", location="NW3").write(store)
    between = dt.datetime.now(dt.UTC)
    Counterparty(name="Acme", location="SW1").write(store)
    assert Counterparty.read(store, name="Acme", entry_time=between).location == "NW3"


def test_read_many_returns_each_requested_id(store):
    acme, other = Counterparty(name="Acme"), Counterparty(name="Other")
    store.write([acme, other])
    assert store.read([acme.object_id, other.object_id]) == {
        acme.object_id: acme,
        other.object_id: other,
    }


# JSON records are validated from the raw JSON, not via duckdb's inferred types


def test_json_discriminated_union_members_with_different_fields(json_store):
    # duckdb's inference merges all members into one struct, giving each the others' fields as nulls
    dog = Kennel(name="dog", pet=Dog(bark="woof"))
    cat = Kennel(name="cat", pet=Cat(lives=9))
    json_store.write([dog, cat])
    assert Kennel.read(json_store, name="dog") == dog
    assert Kennel.read(json_store, name="cat") == cat


def test_json_types_orjson_cannot_encode(json_store):
    kennel = Kennel(name="k", pet=Cat(lives=1), fee=Decimal("12.50"), tags=("x", "y"))
    kennel.write(json_store)
    assert Kennel.read(json_store, name="k") == kennel


# Field type support, by format. Known failures are strict xfails, so fixing one fails the run until
# its mark is removed


class Colour(StrEnum):
    Red = "red"


Json, Parquet = RecordFormat.Json, RecordFormat.Parquet

# label: (field type, value, {format: (failure reason, exception raised)})
FIELD_TYPES = {
    "decimal": (
        Decimal,
        Decimal("12.50"),
        {
            Parquet: (
                "anyschema Decimal has no precision/scale, so values are rescaled",
                ArrowInvalid,
            )
        },
    ),
    "date": (dt.date, dt.date(2026, 1, 2), {}),
    "time": (dt.time, dt.time(3, 4, 5), {}),
    "naive_datetime": (dt.datetime, dt.datetime(2026, 1, 2, 3, 4, 5, 6), {}),
    "aware_datetime": (
        dt.datetime,
        dt.datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=dt.UTC),
        {Parquet: ("Silently loses the time zone", AssertionError)},
    ),
    "timedelta": (
        dt.timedelta,
        dt.timedelta(hours=1, microseconds=5),
        {Parquet: ("Silently reads back a different value", AssertionError)},
    ),
    "bytes": (
        bytes,
        b"\xff\x00",
        {
            Json: (
                "pydantic's default ser_json_bytes='utf8' cannot encode non-UTF-8",
                PydanticSerializationError,
            )
        },
    ),
    "str_enum": (
        Colour,
        Colour.Red,
        {
            Parquet: (
                "narwhals cannot convert an Enum dtype to pyarrow",
                NotImplementedError,
            )
        },
    ),
    "tuple": (tuple[str, ...], ("a", "b"), {}),
    "frozenset": (
        frozenset[str],
        frozenset({"a"}),
        {Parquet: ("anyschema has no parser for sets", NotImplementedError)},
    ),
    "uuid": (
        UUID,
        UUID(int=1),
        {Parquet: ("anyschema has no parser for UUID", NotImplementedError)},
    ),
    "optional_none": (int | None, None, {}),
    "dict": (
        dict[str, int],
        {"a": 1},
        {
            Parquet: (
                "anyschema maps dict to an empty struct, which Parquet rejects",
                ArrowNotImplementedError,
            )
        },
    ),
}


def _field_type_params():
    for label, (typ, value, failures) in FIELD_TYPES.items():
        # Not make_dataclass: it sets annotations after class creation, too late for Identifiable.__init_subclass__
        holder = dataclass(frozen=True, kw_only=True)(
            type(
                f"Holds_{label}",
                (Identifiable,),
                {
                    "__annotations__": {"name": IdField[str], "value": typ},
                    "__module__": __name__,
                },
            )
        )
        registry.register(holder.__name__, holder)

        for record_format in RecordFormat:
            marks = ()
            if (failure := failures.get(record_format)) is not None:
                reason, raises = failure
                marks = pytest.mark.xfail(strict=True, reason=reason, raises=raises)
            yield pytest.param(
                holder,
                value,
                record_format,
                id=f"{label}-{record_format.name}",
                marks=marks,
            )


@pytest.mark.parametrize(
    ("holder", "value", "record_format"), list(_field_type_params())
)
def test_field_type_round_trip(tmp_path, holder, value, record_format):
    store = FSStore(f"file://{tmp_path}", record_format=record_format)
    obj = holder(name="x", value=value)
    obj.write(store)

    read = holder.read(store, name="x")
    assert read == obj
    assert type(read.value) is type(
        value
    )  # e.g. a str would compare equal to a StrEnum
