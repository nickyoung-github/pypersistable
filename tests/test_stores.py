"""Tests of every store: see conftest.STORE_KINDS"""

import contextlib
import dataclasses
import datetime as dt
import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from itertools import pairwise
from typing import Annotated, Any, Literal
from uuid import UUID

import pytest
from models import (
    Cat,
    Counterparty,
    CustomerBook,
    Deal,
    Dog,
    Evolving,
    EvolvingBefore,
    EvolvingNoDefault,
    EvolvingRetyped,
    HoldsIdNode,
    IdNode,
    Kennel,
    ModelDeal,
    Node,
    Option,
    PlainUnion,
    TradingBook,
    loopback_ok,
    registry,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NaiveDatetime,
    PrivateAttr,
    ValidationError,
)

from pypersistable import (
    DiscriminatedUnion,
    Identifiable,
    IdField,
    IncompatibleSchemaError,
    Reference,
    UnknownObjectError,
    UnsupportedTypeError,
    WriteConflictError,
)

SQL = ("SQLite", "PostgreSQL")


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


# Batching: references are de-duplicated and checked on write, and resolved a level at a time on read


@pytest.fixture
def spy(store, monkeypatch):
    """Records, per call, the number of ids queried by type (reads) and of records written per batch (writes)"""
    calls = {"reads": [], "writes": []}
    read, write = store._read, store._write

    def spy_read(by_type, *args):
        calls["reads"].append({t.__name__: len(ids) for t, ids in by_type.items()})
        return read(by_type, *args)

    def spy_write(batches, *args):
        calls["writes"].append([(t.__name__, len(records)) for t, records in batches])
        return write(batches, *args)

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
    (batches,) = spy["writes"]
    # Referenced objects first
    assert sorted(batches[:2]) == [("Counterparty", 1), ("TradingBook", 1)]
    assert batches[2] == ("Deal", 3)


def test_unchanged_objects_and_references_are_not_rewritten(store, spy):
    store.write(shared_deals())
    spy["reads"].clear()
    store.write(shared_deals())
    assert len(spy["writes"]) == 1
    # One read to check: references are compared by id, so are not resolved
    assert spy["reads"] == [{"Deal": 3, "Counterparty": 1, "TradingBook": 1}]


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
    assert spy["writes"][-1] == [("Counterparty", 1)]
    assert Deal.read(store, deal_id="D0") == deal


@pytest.fixture
def strict_store(make_store):
    return make_store(require_load_of_existing=True)


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


def test_equal_instances(strict_store):
    Node(name="a").write(strict_store)
    read = Node.read(strict_store, name="a")
    read.next = Node(name="b")
    anew = Node(name="a", next=Node(name="b"))

    # anew was not read, but read, which is equal, is of the latest version
    strict_store.write([anew, read])
    assert anew.object_entry_time == read.object_entry_time is not None

    strict_store.delete([anew, read])
    assert anew.object_entry_time == read.object_entry_time is not None
    with pytest.raises(UnknownObjectError):
        Node.read(strict_store, name="a")


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


def test_delete(store):
    node = Node(name="a")
    node.write(store)
    before = dt.datetime.now(dt.UTC)
    node.del_(store, effective_time=before + dt.timedelta(days=1))

    with pytest.raises(UnknownObjectError):
        Node.read(store, name="a")
    assert store.read([node.object_id]) == {}

    # Its data is kept: found as of before it was deleted, by either time
    assert Node.read(store, name="a", entry_time=before) == node
    assert Node.read(store, name="a", effective_time=before) == node


def test_delete_by_id_fields(strict_store):
    node = Node(name="a")
    node.write(strict_store)
    # Reads it first, so it may be deleted from a store requiring a load of existing objects
    Node.delete(strict_store, name="a")
    with pytest.raises(UnknownObjectError):
        Node.read(strict_store, name="a")
    with pytest.raises(UnknownObjectError):
        Node.delete(strict_store, name="a")


def test_delete_ignores_unpersisted_and_deleted(store, spy):
    Node(name="a").del_(store)
    node = Node(name="b")
    node.write(store)
    node.del_(store)
    node.del_(store)
    assert spy["writes"] == [[("Node", 1)], [("Node", 1)]]


def test_write_restores_deleted(store):
    node = Node(name="a")
    node.write(store)
    node.del_(store)
    node.write(store)  # Unchanged, but restored
    assert Node.read(store, name="a") == node

    # Also by an object constructed anew
    Node.read(store, name="a").del_(store)
    restored = Node(name="a", next=Node(name="b"))
    restored.write(store)
    assert Node.read(store, name="a") == restored


def test_delete_out_of_date_raises(store):
    Node(name="a").write(store)
    mine, theirs = Node.read(store, name="a"), Node.read(store, name="a")
    theirs.next = Node(name="b")
    theirs.write(store)

    with pytest.raises(WriteConflictError, match="Node/.* was written at"):
        mine.del_(store)
    with pytest.warns(UserWarning, match="Overwriting"):
        mine.del_(store, force_overwrite=True)
    with pytest.raises(UnknownObjectError):
        Node.read(store, name="a")


def test_reference_to_deleted_resolves(store):
    tail = Node(name="b")
    head = Node(name="a", next=tail)
    head.write(store)
    tail.del_(store)

    with pytest.raises(UnknownObjectError):
        Node.read(store, name="b")
    assert Node.read(store, name="a") == head


def test_write_restores_deleted_reference(store, spy):
    tail = Node(name="b")
    head = Node(name="a", next=tail)
    head.write(store)
    tail.del_(store)

    # Unchanged itself, but writes the deleted object it references, restoring it
    head.write(store)
    assert spy["writes"][-1] == [("Node", 1)]
    assert Node.read(store, name="b") == tail


def test_require_load_of_existing_delete(strict_store):
    Node(name="a").write(strict_store)
    with pytest.raises(WriteConflictError, match="was not read"):
        Node(name="a").del_(strict_store)

    Node.read(strict_store, name="a").del_(strict_store)
    restored = Node(name="a", next=Node(name="b"))
    restored.write(strict_store)  # A deleted object may be restored without reading it
    assert Node.read(strict_store, name="a") == restored


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

    assert mine.next is not None
    mine.next.next = Node(name="d")
    with pytest.raises(WriteConflictError, match="Node/.* was written at"):
        mine.write(store)


def test_changed_object_is_rewritten(store, spy):
    deal = shared_deals(1)[0]
    store.write([deal])
    store.write([dataclasses.replace(deal, backup=Counterparty(name="Backup"))])
    assert spy["writes"][-1] == [("Counterparty", 1), ("Deal", 1)]


@dataclass(kw_only=True)
class Left(Identifiable):
    name: IdField[str]
    right: "Reference[Right] | None" = None


@dataclass(kw_only=True)
class Right(Identifiable):
    name: IdField[str]
    left: Reference[Left] | None = None


@dataclass(kw_only=True)
class Located(Identifiable):
    name: IdField[str]
    counterparties: tuple[Reference[Counterparty], ...]
    by_role: dict[str, Reference[Counterparty]]
    locations: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self):
        # Of the objects referenced: when reading, run once they are loaded
        counterparties = (*self.counterparties, *self.by_role.values())
        self.locations = tuple(c.location for c in counterparties)


@dataclass(kw_only=True)
class LocatedBase:
    """Not Persistable"""

    counterparty: Reference[Counterparty]
    location: str = field(init=False, default="")

    def __post_init__(self):
        self.location = self.counterparty.location


@dataclass(kw_only=True)
class InheritsLocated(LocatedBase, Identifiable):
    name: IdField[str]


class ModelLocated(Identifiable, BaseModel):
    name: IdField[str]
    counterparty: Reference[Counterparty]
    location: str = ""

    def model_post_init(self, context: Any):
        self.location = self.counterparty.location


class ModelLocatedPrivate(ModelLocated):
    # pydantic wraps model_post_init, to initialise them
    _cache: dict = PrivateAttr(default_factory=dict)


registry.register("Left", Left)
registry.register("Right", Right)
for _typ in (Located, InheritsLocated, ModelLocated, ModelLocatedPrivate):
    registry.register(_typ.__name__, _typ)


def test_written_in_batches_referenced_first(store, spy):
    # Of one type, so one batch
    Node(name="a", next=Node(name="b", next=Node(name="c"))).write(store)
    # Of types referencing each other: a Left after the Right after the Left
    left = Left(name="a", right=Right(name="b", left=Left(name="c")))
    left.write(store)

    assert spy["writes"] == [[("Node", 3)], [("Left", 1), ("Right", 1), ("Left", 1)]]
    assert Left.read(store, name="a") == left


def test_post_init_after_references_resolved(store):
    acme = Counterparty(name="Acme", location="NW3")
    other = Counterparty(name="Other", location="SW1")
    Located(name="a", counterparties=(acme,), by_role={"agent": other}).write(store)
    assert Located.read(store, name="a").locations == ("NW3", "SW1")


@pytest.mark.parametrize(
    "typ",
    [InheritsLocated, ModelLocated, ModelLocatedPrivate],
    ids=lambda t: t.__name__,
)
def test_post_init_after_references_resolved_of(store, typ):
    # __post_init__ inherited from a base not Persistable, and BaseModel's model_post_init
    typ(name="a", counterparty=Counterparty(name="Acme", location="NW3")).write(store)
    read = typ.read(store, name="a")
    assert read.location == "NW3"
    if typ is ModelLocatedPrivate:
        assert read._cache == {}


def test_object_effective_later_is_rewritten(store):
    now, year = dt.datetime.now(dt.UTC), dt.timedelta(days=365)
    Node(name="b").write(store, effective_time=now + year)
    # Unchanged, but rewritten: else not found as of a's effective time
    node = Node(name="a", next=Node(name="b"))
    node.write(store, effective_time=now - year)
    assert Node.read(store, name="a", effective_time=now) == node


@loopback_ok
def test_concurrent_async_reads(store):
    counterparties = [Counterparty(name=f"C{i}") for i in range(10)]
    store.write(counterparties)
    now = dt.datetime.now(dt.UTC)
    # Different times: not batched, so one store read each, run concurrently
    reads = [
        Counterparty.read(
            store,
            name=c.name,
            entry_time=now + dt.timedelta(microseconds=i),
            is_async=True,
        )
        for i, c in enumerate(counterparties)
    ]
    assert store.read_all(*reads) == counterparties


def test_user_unknown(store, monkeypatch):
    def no_user():
        raise OSError("No username set in the environment")

    monkeypatch.setattr("getpass.getuser", no_user)
    Node(name="a").write(store)
    assert Node.read(store, name="a") == Node(name="a")


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


def test_references_read_once(store, spy):
    nodes = [Node(name=n) for n in "abcd"]
    for node, next_node in pairwise(nodes):
        node.next = next_node
    nodes[0].write(store)
    spy["reads"].clear()

    # All read together, including those they reference, so read once
    assert store.read([n.object_id for n in nodes]) == {n.object_id: n for n in nodes}
    assert spy["reads"] == [{"Node": 4}]
    # Otherwise a level at a time
    spy["reads"].clear()
    assert Node.read(store, name="a") == nodes[0]
    assert spy["reads"] == [{"Node": 1}] * 4


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


def test_self_reference(store):
    head = Node(name="a", next=Node(name="b", next=Node(name="c")))
    head.write(store)
    assert Node.read(store, name="a") == head


def test_referenced_id_field_reference_cycle_raises(store):
    a, b = IdNode("a"), IdNode("b")
    a.parent, b.parent = b, a
    # a's id is computed serialising the object referencing it: raised as is, not wrapped by pydantic
    with pytest.raises(
        ValueError, match="^Reference cycle in id fields: IdNode -> IdNode -> IdNode$"
    ):
        HoldsIdNode(name="h", node=a).write(store)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_reference_cycle_raises(store, spy, length):
    nodes = [Node(name=str(i)) for i in range(length)]
    for node, next_node in zip(nodes, nodes[1:] + nodes[:1], strict=True):
        node.next = next_node

    with pytest.raises(ValueError, match="Reference cycle: Node/"):
        store.write(nodes[:1])
    assert spy["writes"] == []


def test_different_objects_with_the_same_id_raise(store, spy):
    deal = Deal(
        deal_id="D",
        counterparty=Counterparty(name="Acme", location="NW3"),
        book=TradingBook(name="B"),
        backup=Counterparty(name="Acme", location="SW1"),
    )
    with pytest.raises(
        ValueError, match="Different objects with the same id: Counterparty/"
    ):
        store.write([deal])
    assert spy["writes"] == []

    # Equal objects are fine
    Deal(**{**vars(deal), "backup": Counterparty(name="Acme", location="NW3")}).write(
        store
    )


def test_plain_union(store, store_kind):
    plain = PlainUnion(name="p", value="text")
    if store_kind == "Parquet":
        # Persisted with its arrow type
        with pytest.raises(UnsupportedTypeError, match=r"\. Use DiscriminatedUnion"):
            plain.write(store)
    else:
        # For SQL, as JSON
        plain.write(store)
        assert PlainUnion.read(store, name="p") == plain


# Field type support, by store kind


class Colour(StrEnum):
    Red = "red"


class Shape(Enum):
    Square = "square"


class Size(IntEnum):
    Large = 3


naive = dt.datetime(2026, 1, 2, 3, 4, 5, 6)

# label: (field type, value, {store kind: exception raised on write})
FIELD_TYPES = {
    "decimal": (Decimal, Decimal("12.50"), {}),
    "decimal_constrained": (
        Annotated[Decimal, Field(max_digits=10, decimal_places=2)],
        Decimal("12.50"),
        {},
    ),
    "decimal_wide": (
        Annotated[Decimal, Field(max_digits=30, decimal_places=10)],
        Decimal("12345678901234567890.0123456789"),
        {},
    ),
    "decimal_unscaled": (
        Annotated[Decimal, Field(max_digits=5)],
        Decimal(12345),
        # Of no fixed scale. For SQL, as JSON
        {"Parquet": UnsupportedTypeError},
    ),
    "date": (dt.date, dt.date(2026, 1, 2), {}),
    "time": (dt.time, dt.time(3, 4, 5), {}),
    "aware_time": (
        dt.time,
        dt.time(3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=2))),
        # Its time zone would be lost: Arrow's times have none
        dict.fromkeys(("Parquet", *SQL), ValueError),
    ),
    "naive_datetime": (
        dt.datetime,
        naive,
        # A plain datetime is stored as UTC: use NaiveDatetime
        dict.fromkeys(("Parquet", *SQL), ValueError),
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
    # JSON has no infinities (nor NaN: see test_nan)
    "infinity": (float, float("inf"), dict.fromkeys(("Json", *SQL), ValueError)),
    "nested_infinity": (
        dict[Colour, list[float]],
        {Colour.Red: [float("-inf")]},
        dict.fromkeys(("Json", *SQL), ValueError),
    ),
    # Persisted by name, but validated by alias, unless by name too
    "aliased": (Annotated[int, Field(alias="v")], 1, {}),
    # Members not persisted as objects: tagged, so a date is not read as a str
    "union_of_date_str": (DiscriminatedUnion[dt.date, str], dt.date(2026, 1, 2), {}),
    "union_of_enum_int": (DiscriminatedUnion[Shape, int], Shape.Square, {}),
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
        yield pytest.param(holder, value, failures, id=label)


@loopback_ok
@pytest.mark.parametrize(("holder", "value", "failures"), list(_field_type_params()))
def test_field_type_round_trip(make_store, store_kind, holder, value, failures):
    store = make_store()
    obj = holder(name="x", value=value)
    if (raises := failures.get(store_kind)) is not None:
        with pytest.raises(raises):
            obj.write(store)
        return

    obj.write(store)

    read = holder.read(store, name="x")
    assert read == obj
    # e.g. a str would compare equal to a StrEnum
    assert type(read.value) is type(value)
    # Unchanged: compared with its persisted schema by a new store, then not rewritten
    read.write(make_store())
    assert read.object_entry_time == obj.object_entry_time


def test_nan(store, store_kind):
    option = Option(ticker="X", strike=float("nan"))
    if store_kind != "Parquet":
        # JSON has no NaN: pydantic would write null
        with pytest.raises(ValueError, match="holds NaN or an infinity"):
            option.write(store)
        assert store.read([option.object_id]) == {}
        return

    option.write(store)
    # Not equal to itself, so not compared
    assert math.isnan(store.read([option.object_id])[option.object_id].strike)


# Schema evolution: each type is another version of the one before, as if its code had been changed


@dataclass
class StrictInner:
    __pydantic_config__ = ConfigDict(extra="forbid")
    x: int
    y: int = 0


@dataclass
class StrictInnerRemoved:
    __pydantic_config__ = ConfigDict(extra="forbid")
    x: int


@dataclass
class HoldsStrict(Identifiable):
    name: IdField[str]
    inner: StrictInner
    inners: list[StrictInner]


@dataclass
class HoldsStrictRemoved(Identifiable):
    name: IdField[str]
    inner: StrictInnerRemoved
    inners: list[StrictInnerRemoved]


@dataclass
class Part:
    x: int


@dataclass
class PartAdded:
    x: int
    y: int = 0


@dataclass
class HoldsParts(Identifiable):
    name: IdField[str]
    by_key: dict[str, Part]
    parts: list[Part]
    maybe: Part | None = None


@dataclass
class HoldsPartsAdded(Identifiable):
    name: IdField[str]
    by_key: dict[str, PartAdded]
    parts: list[PartAdded]
    maybe: PartAdded | None = None


@dataclass
class Pet:
    x: int
    y: int = 0


@dataclass
class PetRemoved:
    x: int


# A redefinition of Pet, without y: its tag in a DiscriminatedUnion is the same
PetRemoved.__qualname__ = "Pet"


@dataclass
class HoldsPet(Identifiable):
    name: IdField[str]
    pet: DiscriminatedUnion[Pet, int]


@dataclass
class HoldsPetRemoved(Identifiable):
    name: IdField[str]
    pet: DiscriminatedUnion[PetRemoved, int]


@dataclass
class HoldsPetFewer(Identifiable):
    name: IdField[str]
    pet: DiscriminatedUnion[Pet, str]


for _typ in (
    HoldsStrict,
    HoldsStrictRemoved,
    HoldsParts,
    HoldsPartsAdded,
    HoldsPet,
    HoldsPetRemoved,
    HoldsPetFewer,
):
    registry.register(_typ.__name__, _typ)


def test_added_field_defaults(store, redefine):
    EvolvingBefore(name="old").write(store)
    redefine(EvolvingBefore, Evolving)
    new = Evolving(name="new", added=2)
    new.write(store)

    old_id = Evolving(name="old").object_id
    read = store.read([old_id, new.object_id])  # One read of both versions' files
    assert read == {old_id: Evolving(name="old"), new.object_id: new}


def test_added_field_without_default_raises(store, redefine):
    EvolvingBefore(name="old").write(store)
    redefine(EvolvingBefore, EvolvingNoDefault)
    with pytest.raises(IncompatibleSchemaError, match="EvolvingNoDefault cannot read"):
        EvolvingNoDefault(name="new", added=1).write(store)


# Including rewriting the object written: checked before the write reads its current version
@pytest.mark.parametrize("name", ["new", "old"])
def test_changed_field_type_raises(store, store_kind, redefine, name):
    Evolving(name="old", added=1).write(store)
    redefine(Evolving, EvolvingRetyped)
    match = {
        "Parquet": "_data.added is string, but int64",
        **dict.fromkeys(SQL, "added is TEXT, but BIGINT when persisted"),
    }.get(store_kind, "cannot read")
    with pytest.raises(IncompatibleSchemaError, match=match):
        EvolvingRetyped(name=name).write(store)


def test_changed_field_type_raises_reading(store, store_kind, redefine):
    # Read before any write, which would check the schema
    Evolving(name="old", added=1).write(store)
    redefine(Evolving, EvolvingRetyped)
    parquet = store_kind == "Parquet"
    with pytest.raises(
        IncompatibleSchemaError if parquet else ValidationError,
        match="_data.added is string, but int64" if parquet else "added",
    ):
        EvolvingRetyped.read(store, name="old")
    # Again, not having been marked checked
    with pytest.raises(IncompatibleSchemaError if parquet else ValidationError):
        EvolvingRetyped.read(store, name="old")


def define_redefined(value_type: type) -> type:
    # As a notebook might, redefining a class of the same name
    return dataclass(
        type(
            "Redefined",
            (Identifiable,),
            {
                "__annotations__": {"name": IdField[str], "value": value_type},
                "__module__": __name__,
            },
        )
    )


def test_redefined_type_is_checked(store):
    before = define_redefined(int)
    registry.register("Redefined", before, replace=True)
    before(name="a", value=1).write(store)

    after = define_redefined(str)
    registry.register("Redefined", after, replace=True)
    with pytest.raises(IncompatibleSchemaError):
        after(name="a", value="1").write(store)


def test_removed_field_is_ignored(store, store_kind, redefine):
    Evolving(name="old", added=1).write(store)
    # EvolvingBefore, a plain dataclass, ignores extra fields: added may be removed
    redefine(Evolving, EvolvingBefore)
    # Warns (Parquet only) once per file, here when the schema check reads the old object
    with parquet_warns(store_kind, "ignoring _data.added, removed from EvolvingBefore"):
        EvolvingBefore(name="new").write(store)

    assert EvolvingBefore.read(store, name="old") == EvolvingBefore(name="old")


def test_removed_field_raises_if_extra_forbidden(store, store_kind, redefine):
    HoldsStrict(name="old", inner=StrictInner(1), inners=[StrictInner(2)]).write(store)
    redefine(HoldsStrict, HoldsStrictRemoved)
    match = (
        "_data.inner.y has been removed; _data.inners.item.y has been removed"
        if store_kind == "Parquet"
        else "cannot read"
    )
    with pytest.raises(IncompatibleSchemaError, match=match):
        HoldsStrictRemoved(name="new", inner=StrictInnerRemoved(1), inners=[]).write(
            store
        )


def parquet_warns(store_kind: str, match: str):
    # Ignored fields are warned of reading Parquet only
    return (
        pytest.warns(UserWarning, match=match)
        if store_kind == "Parquet"
        else contextlib.nullcontext()
    )


def test_field_added_in_collections_defaults(store, redefine):
    HoldsParts(name="old", by_key={"a": Part(1)}, parts=[Part(2)], maybe=Part(3)).write(
        store
    )
    redefine(HoldsParts, HoldsPartsAdded)
    HoldsPartsAdded(name="new", by_key={}, parts=[]).write(store)

    assert HoldsPartsAdded.read(store, name="old") == HoldsPartsAdded(
        name="old", by_key={"a": PartAdded(1)}, parts=[PartAdded(2)], maybe=PartAdded(3)
    )


def test_field_removed_in_collections_is_ignored(store, store_kind, redefine):
    HoldsPartsAdded(
        name="old",
        by_key={"a": PartAdded(1, 1)},
        parts=[PartAdded(2, 2)],
        maybe=PartAdded(3, 3),
    ).write(store)
    redefine(HoldsPartsAdded, HoldsParts)
    with parquet_warns(
        store_kind, "ignoring _data.by_key.value.y, _data.parts.item.y, _data.maybe.y"
    ):
        HoldsParts(name="new", by_key={}, parts=[]).write(store)

    assert HoldsParts.read(store, name="old") == HoldsParts(
        name="old", by_key={"a": Part(1)}, parts=[Part(2)], maybe=Part(3)
    )


def test_field_removed_in_union_member_is_ignored(store, store_kind, redefine):
    HoldsPet(name="old", pet=Pet(1, 2)).write(store)
    redefine(HoldsPet, HoldsPetRemoved)
    with parquet_warns(store_kind, "ignoring _data.pet.*Pet.y"):
        HoldsPetRemoved(name="new", pet=1).write(store)

    assert HoldsPetRemoved.read(store, name="old") == HoldsPetRemoved(
        name="old", pet=PetRemoved(1)
    )


def test_removed_union_member_raises(store, store_kind, redefine):
    HoldsPet(name="old", pet=1).write(store)
    redefine(HoldsPet, HoldsPetFewer)
    match = (
        "_data.pet.builtins:int has been removed"
        if store_kind == "Parquet"
        else "cannot read"
    )
    with pytest.raises(IncompatibleSchemaError, match=match):
        HoldsPetFewer(name="new", pet="a").write(store)
