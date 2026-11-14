import dataclasses
import datetime as dt
import os
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from ipaddress import IPv4Address
from typing import Annotated, Literal
from uuid import UUID

import duckdb
import pyarrow as pa
import pytest
from models import (
    Cat,
    Counterparty,
    CustomerBook,
    Deal,
    Dog,
    Kennel,
    ModelDeal,
    Node,
    PlainUnion,
    TradingBook,
    registry,
)
from pydantic import Field, NaiveDatetime

from pypersistable import (
    FSStore,
    Identifiable,
    IdField,
    RecordFormat,
    UnknownObjectError,
    UnsupportedTypeError,
    WriteConflictError,
    persistable_arrow_converter,
)


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


def moved_acme_deal() -> Deal:
    moved = Counterparty(name="Acme", location="SW1")
    return Deal(
        deal_id="D0", counterparty=moved, book=TradingBook(name="Book"), others=(moved,)
    )


def test_changed_reference_is_written(store, spy):
    store.write(shared_deals(1))
    deal = moved_acme_deal()
    store.write([deal])

    # Only the counterparty: the deal's record, referencing it by id, is unchanged
    assert spy["writes"][-1] == {"Counterparty": 1}
    assert Deal.read(store, deal_id="D0") == deal


@pytest.fixture
def strict_store(tmp_path, store) -> FSStore:
    return FSStore(
        f"file://{tmp_path}",
        record_format=store.record_format,
        require_load_of_existing=True,
    )


def test_require_load_of_existing(strict_store):
    strict_store.write(shared_deals(1))
    with pytest.raises(WriteConflictError, match="Counterparty/.* was not read"):
        strict_store.write([moved_acme_deal()])

    # New, unchanged or read objects may be written
    Node(name="a").write(strict_store)
    Node(name="a").write(strict_store)
    read = Node.read(strict_store, name="a")
    read.next = Node(name="b")
    read.write(strict_store)
    assert Node.read(strict_store, name="a") == read


def test_force_overwrite_unread(strict_store):
    strict_store.write(shared_deals(1))
    deal = moved_acme_deal()
    with pytest.warns(UserWarning, match="Overwriting: Counterparty/.* was not read"):
        strict_store.write([deal], force_overwrite=True)
    assert Deal.read(strict_store, deal_id="D0") == deal


def test_force_overwrite_out_of_date(store):
    Node(name="a").write(store)
    mine, theirs = Node.read(store, name="a"), Node.read(store, name="a")
    theirs.next = Node(name="b")
    theirs.write(store)

    mine.next = Node(name="c")
    with pytest.warns(UserWarning, match="Overwriting: Node/.* was written at"):
        mine.write(store, force_overwrite=True)
    assert Node.read(store, name="a") == mine


def test_object_entry_time(store):
    node = Node(name="a")
    assert node.object_entry_time is None

    node.write(store)
    written = node.object_entry_time
    assert written is not None
    assert Node.read(store, name="a").object_entry_time == written

    node.write(store)  # Unchanged: not written again
    assert node.object_entry_time == written


def test_stale_write_raises(store):
    Node(name="a").write(store)
    mine, theirs = Node.read(store, name="a"), Node.read(store, name="a")

    theirs.next = Node(name="b")
    theirs.write(store)
    theirs.next = Node(name="c")
    theirs.write(store)  # Its own version is the latest

    mine.next = Node(name="d")
    with pytest.raises(WriteConflictError, match="was written at"):
        mine.write(store)


def test_stale_reference_raises(store):
    head = Node(name="a", next=Node(name="b"))
    head.write(store)
    mine = Node.read(store, name="a")

    theirs = Node.read(store, name="b")
    theirs.next = Node(name="c")
    theirs.write(store)

    mine.next.next = Node(name="d")
    with pytest.raises(WriteConflictError, match="Node/.* was written at"):
        mine.write(store)


def test_changed_object_is_rewritten(store, spy):
    deal = shared_deals(1)[0]
    store.write([deal])
    store.write([dataclasses.replace(deal, backup=Counterparty(name="Backup"))])
    assert spy["writes"][-1] == {"Deal": 1, "Counterparty": 1}


def test_references_resolved_in_one_read_per_level(store, spy):
    deals = shared_deals(10)
    store.write(deals)
    spy["reads"].clear()

    assert store.read([d.object_id for d in deals]) == {d.object_id: d for d in deals}
    assert spy["reads"] == [{"Deal": 10}, {"Counterparty": 1, "TradingBook": 1}]


def test_references_loaded_in_the_same_read_are_not_read_again(store, spy):
    deal = shared_deals(1)[0]
    store.write([deal])
    spy["reads"].clear()

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


def test_self_reference(store):
    head = Node(name="a", next=Node(name="b", next=Node(name="c")))
    head.write(store)
    assert Node.read(store, name="a") == head


@pytest.mark.parametrize("length", [1, 2, 3])
def test_reference_cycle_raises(store, tmp_path, length):
    nodes = [Node(name=str(i)) for i in range(length)]
    for node, next_node in zip(nodes, nodes[1:] + nodes[:1], strict=True):
        node.next = next_node

    with pytest.raises(ValueError, match="Reference cycle: Node/"):
        store.write(nodes[:1])
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]  # Nothing written


def test_plain_union(store):
    plain = PlainUnion(name="p", value="text")
    if store.record_format is RecordFormat.Parquet:
        with pytest.raises(UnsupportedTypeError, match=r"\. Use DiscriminatedUnion"):
            plain.write(store)
    else:
        plain.write(store)
        assert PlainUnion.read(store, name="p") == plain


@pytest.mark.parametrize(
    ("hint", "suggestion"),
    [
        (int | str, r"\. Use DiscriminatedUnion"),
        (Counterparty | TradingBook | None, "Use Reference.*or DiscriminatedUnion"),
        (Counterparty | int, r"\. Use DiscriminatedUnion"),
    ],
    ids=str,
)
def test_union_suggestion(hint, suggestion):
    with pytest.raises(UnsupportedTypeError, match=suggestion):
        persistable_arrow_converter().arrow_type(hint)


# Field type support, by format


class Colour(StrEnum):
    Red = "red"


class Shape(Enum):
    Square = "square"


class Size(IntEnum):
    Large = 3


Json, Parquet = RecordFormat.Json, RecordFormat.Parquet
naive = dt.datetime(2026, 1, 2, 3, 4, 5, 6)

# label: (field type, value, {format: exception raised on write})
FIELD_TYPES = {
    "decimal": (Decimal, Decimal("12.50"), {}),
    "decimal_constrained": (
        Annotated[Decimal, Field(max_digits=10, decimal_places=2)],
        Decimal("12.50"),
        {},
    ),
    "date": (dt.date, dt.date(2026, 1, 2), {}),
    "time": (dt.time, dt.time(3, 4, 5), {}),
    "naive_datetime": (
        dt.datetime,
        naive,
        {Parquet: ValueError},  # A plain datetime is stored as UTC: use NaiveDatetime
    ),
    "naive_datetime_annotated": (NaiveDatetime, naive, {}),
    "aware_datetime": (dt.datetime, naive.replace(tzinfo=dt.UTC), {}),
    "aware_datetime_not_utc": (
        dt.datetime,
        naive.replace(tzinfo=dt.timezone(dt.timedelta(hours=2))),
        {},
    ),
    "timedelta": (dt.timedelta, dt.timedelta(hours=1, microseconds=5), {}),
    "bytes": (bytes, b"\xff\x00", {}),
    "str_enum": (Colour, Colour.Red, {}),
    "enum": (Shape, Shape.Square, {}),
    "int_enum": (Size, Size.Large, {}),
    "literal": (Literal["a", "b"], "b", {}),
    "tuple": (tuple[str, ...], ("a", "b"), {}),
    "set": (set[int], {1, 2}, {}),
    "frozenset": (frozenset[str], frozenset({"a"}), {}),
    "uuid": (UUID, UUID(int=1), {}),
    "optional_none": (int | None, None, {}),
    "dict": (dict[str, int], {"a": 1}, {}),
    "dict_of_lists": (dict[Colour, list[float]], {Colour.Red: [1.5]}, {}),
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
            yield pytest.param(
                holder,
                value,
                record_format,
                failures.get(record_format),
                id=f"{label}-{record_format.name}",
            )


@pytest.mark.parametrize(
    ("holder", "value", "record_format", "raises"), list(_field_type_params())
)
def test_field_type_round_trip(tmp_path, holder, value, record_format, raises):
    store = FSStore(f"file://{tmp_path}", record_format=record_format)
    obj = holder(name="x", value=value)
    if raises is not None:
        with pytest.raises(raises):
            obj.write(store)
        return

    obj.write(store)

    read = holder.read(store, name="x")
    assert read == obj
    # e.g. a str would compare equal to a StrEnum
    assert type(read.value) is type(value)


@dataclass
class Host(Identifiable):
    name: IdField[str]
    address: IPv4Address


registry.register(Host.__name__, Host)


@dataclass
class Evolving(Identifiable):
    name: IdField[str]
    added: int = 7  # Added after the first version was written


registry.register(Evolving.__name__, Evolving)


def test_parquet_added_field_defaults(tmp_path):
    url = f"file://{tmp_path}"
    # Writes Evolving as it was before the added field
    before = persistable_arrow_converter()
    before.add_struct(lambda t: {"name": IdField[str]} if t is Evolving else None)
    Evolving(name="old", added=1).write(FSStore(url, arrow_converter=before))

    store = FSStore(url)
    new = Evolving(name="new", added=2)
    new.write(store)

    # One read of both files
    old_id = Evolving(name="old").object_id
    read = store.read([old_id, new.object_id])
    assert read == {old_id: Evolving(name="old"), new.object_id: new}


def test_parquet_custom_type(tmp_path):
    url = f"file://{tmp_path}"
    with pytest.raises(UnsupportedTypeError, match="IPv4Address"):
        Host(name="h", address=IPv4Address("10.0.0.1")).write(FSStore(url))

    converter = persistable_arrow_converter()
    converter.add_type(IPv4Address, pa.string(), str)
    store = FSStore(url, arrow_converter=converter)
    host = Host(name="h", address=IPv4Address("10.0.0.1"))
    host.write(store)
    assert Host.read(store, name="h") == host
