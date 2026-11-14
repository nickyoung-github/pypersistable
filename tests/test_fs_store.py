import dataclasses
import datetime as dt
import os
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

import duckdb
import pytest
from anyschema.exceptions import UnsupportedDTypeError
from models import (
    Cat,
    Counterparty,
    CustomerBook,
    Deal,
    Dog,
    Kennel,
    ModelDeal,
    PlainUnion,
    TradingBook,
    registry,
)
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


@pytest.mark.parametrize("pet", [Dog(bark="woof"), Cat(lives=9), 3], ids=type)
def test_discriminated_union_round_trip(store, pet):
    # JSON: validated from the raw JSON, not duckdb's inferred schema, which merges the members into one struct
    # Parquet: persisted as a struct of the tag and a field per member
    kennel = Kennel(name="k", pet=pet)
    kennel.write(store)
    assert Kennel.read(store, name="k") == kennel


@pytest.mark.parametrize(
    "book", [TradingBook(name="Book"), Counterparty(name="Book")], ids=type
)
def test_references_round_trip(store, book):
    acme, backup, other = (
        Counterparty(name=n, location=n) for n in ("Acme", "Backup", "Other")
    )
    deal = Deal(
        deal_id="D", counterparty=acme, book=book, backup=backup, others=(other, acme)
    )
    deal.write(store)

    assert Deal.read(store, deal_id="D") == deal
    # References are persisted separately, so may be read in their own right
    assert Counterparty.read(store, name="Other") == other
    assert type(book).read(store, name="Book") == book


def test_optional_reference_none_round_trip(store):
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    deal.write(store)
    assert Deal.read(store, deal_id="D") == deal


def test_base_model_with_reference_and_discriminated_union(store):
    deal = ModelDeal(
        deal_id="D", counterparty=Counterparty(name="Acme"), pet=Dog(bark="woof")
    )
    deal.write(store)
    assert ModelDeal.read(store, deal_id="D") == deal


def test_json_by_value_round_trip(json_store):
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    json_store.write([deal], by_reference=False)

    assert Deal.read(json_store, deal_id="D") == deal
    with pytest.raises(UnknownObjectError):
        Counterparty.read(json_store, name="Acme")  # Embedded, not persisted separately


# Batching: references are de-duplicated and checked on write, and resolved a level at a time on read


@pytest.fixture
def spy(store, monkeypatch):
    """Records, per call, the number of ids queried by type (reads) and records written by type (writes)"""
    calls = {"reads": [], "writes": []}
    read, write = store._read, store._write

    def spy_read(by_type, *args):
        calls["reads"].append({t.__name__: len(ids) for t, ids in by_type.items()})
        return read(by_type, *args)

    def spy_write(by_type, *args):
        calls["writes"].append(
            {t.__name__: len(records) for t, records in by_type.items()}
        )
        return write(by_type, *args)

    monkeypatch.setattr(store, "_read", spy_read)
    monkeypatch.setattr(store, "_write", spy_write)
    return calls


def shared_deals(count: int = 3) -> list[Deal]:
    acme, book = Counterparty(name="Acme"), TradingBook(name="Book")
    return [
        Deal(deal_id=f"D{i}", counterparty=acme, book=book, others=(acme,))
        for i in range(count)
    ]


def test_write_accepts_a_generator(store):
    deals = shared_deals()
    store.write(d for d in deals)
    assert store.read([d.object_id for d in deals]) == {d.object_id: d for d in deals}


def test_shared_references_are_written_once(store, spy):
    store.write(shared_deals())
    assert spy["writes"] == [{"Deal": 3, "Counterparty": 1, "TradingBook": 1}]


def test_unchanged_objects_and_references_are_not_rewritten(store, spy):
    store.write(shared_deals())
    store.write(shared_deals())
    assert len(spy["writes"]) == 1


def test_changed_reference_is_rewritten(store, spy):
    store.write(shared_deals(1))
    deal = Deal(
        deal_id="D0",
        counterparty=Counterparty(name="Acme", location="SW1"),
        book=TradingBook(name="Book"),
        others=(Counterparty(name="Acme", location="SW1"),),
    )
    store.write([deal])

    # Only the counterparty: the deal's record, referencing it by id, is unchanged
    assert spy["writes"][-1] == {"Counterparty": 1}
    assert Deal.read(store, deal_id="D0") == deal


def test_changed_object_is_rewritten(store, spy):
    deal = shared_deals(1)[0]
    store.write([deal])
    store.write([dataclasses.replace(deal, backup=Counterparty(name="Backup"))])
    assert spy["writes"][-1] == {"Deal": 1, "Counterparty": 1}


def test_check_false_writes_everything_once(store, spy):
    store.write(shared_deals())
    store.write(shared_deals(), check=False)
    assert spy["writes"][-1] == {"Deal": 3, "Counterparty": 1, "TradingBook": 1}


def test_references_resolved_in_one_read_per_level(store, spy):
    deals = shared_deals(10)
    store.write(deals, check=False)

    assert store.read([d.object_id for d in deals]) == {d.object_id: d for d in deals}
    assert spy["reads"] == [{"Deal": 10}, {"Counterparty": 1, "TradingBook": 1}]


def test_references_loaded_in_the_same_read_are_not_read_again(store, spy):
    deal = shared_deals(1)[0]
    store.write([deal], check=False)

    store.read([deal.object_id, deal.counterparty.object_id, deal.book.object_id])
    assert spy["reads"] == [{"Deal": 1, "Counterparty": 1, "TradingBook": 1}]


def test_references_resolved_as_of_the_same_time(store):
    deal = Deal(
        deal_id="D",
        counterparty=Counterparty(name="Acme", location="NW3"),
        book=TradingBook(name="B"),
    )
    deal.write(store)
    before_move = dt.datetime.now(dt.UTC)
    Counterparty(name="Acme", location="SW1").write(store)

    assert (
        Deal.read(store, deal_id="D", entry_time=before_move).counterparty.location
        == "NW3"
    )
    assert Deal.read(store, deal_id="D").counterparty.location == "SW1"


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root can read files without read permission"
)
def test_unreadable_file_raises(store, tmp_path):
    Counterparty(name="Acme").write(store)
    for path in (tmp_path / "Counterparty").iterdir():
        path.chmod(0)

    try:
        with pytest.raises(duckdb.IOException):
            Counterparty.read(store, name="Acme")
    finally:
        for path in (tmp_path / "Counterparty").iterdir():
            path.chmod(0o644)


def test_dangling_reference_raises(tmp_path):
    store = FSStore(f"file://{tmp_path}", record_format=RecordFormat.Json)
    deal = shared_deals(1)[0]
    store.write([deal])
    for path in (tmp_path / "Counterparty").iterdir():
        path.unlink()

    with pytest.raises(RuntimeError, match="Failed to load references"):
        Deal.read(store, deal_id="D0")


def test_parquet_by_value_raises(tmp_path):
    store = FSStore(f"file://{tmp_path}", record_format=RecordFormat.Parquet)
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    with pytest.raises(ValueError, match="by_reference=False is not supported"):
        store.write([deal], by_reference=False)


def test_plain_union(store):
    plain = PlainUnion(name="p", value="text")
    if store.record_format is RecordFormat.Parquet:
        with pytest.raises(
            UnsupportedDTypeError, match="Use Reference.*or DiscriminatedUnion"
        ):
            plain.write(store)
    else:
        plain.write(store)
        assert PlainUnion.read(store, name="p") == plain


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
