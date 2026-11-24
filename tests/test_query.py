"""Queries, of every store supporting them: see conftest.STORE_KINDS"""
# To type checkers, a class's field is of its type, so one which may be None has no fields
# pyright: reportOptionalMemberAccess=false

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

import pytest
import sqlalchemy as sa
from models import registry
from pydantic import BaseModel

from pypersistable import (
    Identifiable,
    IdField,
    PersistableStoreBase,
    Reference,
    UnsupportedTypeError,
    ValidatingDataclassMeta,
)
from pypersistable.query import FieldColumn, col, select


class Sector(StrEnum):
    BANK = "bank"
    FUND = "fund"


@dataclass(frozen=True, kw_only=True)
class QAddress:
    city: str
    country: str
    floor: int | None = None
    opened: dt.date | None = None


@dataclass(frozen=True, kw_only=True)
class QParty(Identifiable):
    name: IdField[str]
    country: str
    rating: int = 3
    sector: Sector = Sector.BANK
    limit: Decimal = Decimal(0)
    since: dt.datetime | None = None
    address: QAddress | None = None
    parent: "Reference[QParty] | None" = None


@dataclass(frozen=True, kw_only=True)
class QBroker(Identifiable):
    name: IdField[str]


@dataclass(frozen=True, kw_only=True)
class QTrade(Identifiable):
    trade_id: IdField[str]
    party: Reference[QParty]
    notional: float
    broker: Reference[QBroker] | None = None


class QModel(Identifiable, BaseModel):
    key: IdField[str]
    value: int = 5
    party: Reference[QParty] | None = None
    either: Reference[QParty | QBroker] | None = None


@dataclass(kw_only=True, slots=True)
class QSlots(Identifiable):
    key: IdField[str]
    value: int = 5


@dataclass(kw_only=True)
class QValidating(Identifiable, metaclass=ValidatingDataclassMeta):
    key: IdField[str]
    value: int = 5


@dataclass(frozen=True, kw_only=True)
class QUnregistered(Identifiable):
    key: IdField[str]
    value: int = 5


for _typ in (QParty, QBroker, QTrade, QModel, QSlots, QValidating):
    registry.register(_typ.__name__, _typ)

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
BE = QParty(
    name="Belgo",
    country="Belgium",
    limit=Decimal("2.5"),
    since=T0,
    address=QAddress(
        city="Ghent", country="Belgium", floor=2, opened=dt.date(2020, 1, 1)
    ),
)
US = QParty(name="USCo", country="USA", rating=5, sector=Sector.FUND, parent=BE)
UK = QParty(name="UKCo", country="UK", parent=US, address=QAddress(city="Leeds", country="UK"))  # fmt: skip
T1, T2, T3 = (
    QTrade(trade_id="T1", party=US, notional=10.0),  # Its party's parent is Belgian
    QTrade(trade_id="T2", party=UK, notional=20.0),  # Its party's parent's parent is
    QTrade(trade_id="T3", party=BE, notional=30.0),  # Its party has no parent
)


@pytest.fixture
def db(store, store_kind):
    if store_kind == "Json":
        pytest.skip("Not supported")
    store.write([T1, T2, T3])
    return store


def names(objects) -> set[str]:
    return {getattr(o, "name", None) or o.trade_id for o in objects}


def test_all(db):
    assert names(db.exec(select(QParty))) == {"Belgo", "USCo", "UKCo"}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (lambda: QParty.country == "USA", {"USCo"}),
        (lambda: QParty.country != "USA", {"Belgo", "UKCo"}),
        (lambda: col(QParty.country).in_(["USA", "UK"]), {"USCo", "UKCo"}),
        (lambda: col(QParty.country).in_([]), set()),
        (lambda: col(QParty.country).startswith("U"), {"USCo", "UKCo"}),
        (lambda: QParty.rating > 3, {"USCo"}),
        (lambda: col(QParty.rating).between(1, 3), {"Belgo", "UKCo"}),
        (lambda: QParty.name == "Belgo", {"Belgo"}),  # An id field
        (lambda: QParty.sector == Sector.FUND, {"USCo"}),
        (lambda: QParty.limit == Decimal("2.5"), {"Belgo"}),
        # At the same instant, in another time zone
        (
            lambda: (
                QParty.since
                == T0.astimezone(dt.timezone(dt.timedelta(hours=5)))
            ),
            {"Belgo"},
        ),
        (lambda: col(QParty.since).is_(None), {"USCo", "UKCo"}),
        (lambda: (QParty.country == "USA") | (QParty.rating == 3), {"Belgo", "USCo", "UKCo"}),
        (lambda: sa.or_(col(QParty.country == "UK"), col(QParty.rating > 3)), {"USCo", "UKCo"}),
        (lambda: ~col(QParty.country == "UK"), {"Belgo", "USCo"}),
        # Of an embedded object
        (lambda: QParty.address.country == "Belgium", {"Belgo"}),
        (lambda: QParty.address.floor == 2, {"Belgo"}),
        (lambda: col(QParty.address.floor).is_(None), {"USCo", "UKCo"}),
        (lambda: col(QParty.address).is_(None), {"USCo"}),
        (lambda: col(QParty.address).is_not(None), {"Belgo", "UKCo"}),
        # A reference
        (lambda: QParty.parent == BE, {"USCo"}),
        (lambda: QParty.parent == BE.object_id, {"USCo"}),
        (lambda: QParty.parent != BE, {"UKCo"}),  # Not Belgo, which has none
        (lambda: col(QParty.parent).in_([BE, US]), {"USCo", "UKCo"}),
        (lambda: col(QParty.parent).in_([BE, None]), {"USCo"}),
        (lambda: col(QParty.parent).is_(None), {"Belgo"}),
        # Of a referenced object
        (lambda: QParty.parent.country == "Belgium", {"USCo"}),
        (lambda: QParty.parent.parent.country == "Belgium", {"UKCo"}),
        (lambda: QParty.parent.address.city == "Ghent", {"USCo"}),
        (lambda: col(QParty.parent.rating).is_(None), {"Belgo"}),
    ],
)  # fmt: skip
def test_where(db, condition, expected):
    assert names(db.exec(select(QParty).where(condition()))) == expected


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (lambda: QTrade.party.country == "USA", {"T1"}),
        (lambda: QTrade.party.parent.country == "Belgium", {"T1"}),
        (lambda: QTrade.party.parent.parent.country == "Belgium", {"T2"}),
        (lambda: (QTrade.party.parent.country == "Belgium") | (QTrade.notional > 25), {"T1", "T3"}),
        (lambda: QTrade.party.parent == BE, {"T1"}),
        # Following a reference once
        (lambda: sa.and_(col(QTrade.party.country == "USA"), col(QTrade.party.rating == 5)), {"T1"}),
        # None of its type are persisted
        (lambda: QTrade.broker.name == "Brokers", set()),
        (lambda: col(QTrade.broker.name).is_(None), {"T1", "T2", "T3"}),
    ],
)  # fmt: skip
def test_where_through_references(db, condition, expected):
    assert names(db.exec(select(QTrade).where(condition()))) == expected


def test_conditions_are_all_met(db):
    query = select(QParty).where(QParty.rating == 3).where(QParty.country == "UK")
    assert names(db.exec(query)) == {"UKCo"}
    assert names(db.exec(select(QParty).where(QParty.rating == 3, QParty.country == "UK"))) == {"UKCo"}  # fmt: skip


def test_objects_are_as_read(db):
    (trade,) = db.exec(select(QTrade).where(QTrade.trade_id == "T2"))
    assert trade == T2
    assert trade.party.parent.parent == BE
    assert trade.object_entry_time is not None


def test_nothing_persisted(store, store_kind):
    if store_kind == "Json":
        pytest.skip("Not supported")
    assert store.exec(select(QParty).where(QParty.country == "USA")) == []


def test_latest_versions(db):
    moved = QParty(name="USCo", country="Canada", rating=5, parent=BE)
    moved.write(db)
    # Not as it was
    assert db.exec(select(QParty).where(QParty.country == "USA")) == []
    assert db.exec(select(QParty).where(QParty.country == "Canada")) == [moved]
    # Nor as the object referenced was
    assert db.exec(select(QTrade).where(QTrade.party.country == "USA")) == []
    assert names(db.exec(select(QTrade).where(QTrade.party.country == "Canada"))) == {"T1"}  # fmt: skip


def test_as_of(db):
    before = dt.datetime.now(dt.UTC)
    QParty(name="USCo", country="Canada", parent=BE).write(db)
    usa = select(QParty).where(QParty.country == "USA")
    assert db.exec(usa, entry_time=before) == [US]
    assert db.exec(usa, effective_time=before) == [US]
    trades = select(QTrade).where(QTrade.party.country == "USA")
    assert db.exec(trades, entry_time=before) == [T1]
    assert db.exec(select(QParty), entry_time=T0) == []


def test_deleted(db):
    UK.del_(db)
    assert names(db.exec(select(QParty))) == {"Belgo", "USCo"}
    # Though referenced objects may be, as read resolves them
    assert db.exec(select(QTrade).where(QTrade.party.country == "UK")) == [T2]


def test_field_added_since_written(db, redefine):
    @dataclass(frozen=True, kw_only=True)
    class Grown(Identifiable):
        name: IdField[str]
        country: str
        region: str = "EMEA"
        address: QAddress | None = None

    Grown._registered()  # As registering it would
    redefine(QParty, Grown)
    # Not persisted, so null, though read as its default
    assert db.exec(select(Grown).where(Grown.region == "EMEA")) == []
    assert len(db.exec(select(Grown).where(col(Grown.region).is_(None)))) == 3


def test_models_slots_and_validating_dataclasses(db):
    objects = [
        QModel(key="a", value=1, party=US, either=US),
        QModel(key="b"),
        QSlots(key="a", value=1),
        QSlots(key="b"),
        QValidating(key="a", value=1),
        QValidating(key="b"),
    ]
    db.write(objects)
    for typ in (QModel, QSlots, QValidating):
        (found,) = db.exec(select(typ).where(typ.value > 3))
        assert (found.key, found.value) == ("b", 5)
    assert [m.key for m in db.exec(select(QModel).where(QModel.party.country == "USA"))] == ["a"]  # fmt: skip

    # A reference to one of several types has no fields, but compares
    assert [m.key for m in db.exec(select(QModel).where(QModel.either == US))] == ["a"]
    assert db.exec(select(QModel).where(QModel.either == BE)) == []

    # Their instances are unchanged
    slots = QSlots(key="c")
    slots.value = 7
    assert slots.value == 7
    validating = QValidating(key="c")
    with pytest.raises(ValueError, match="valid integer"):
        validating.value = "x"  # pyright: ignore[reportAttributeAccessIssue]
    del validating.value
    assert validating.value == 5  # Its default


def test_json_is_not_supported(store, store_kind):
    if store_kind != "Json":
        pytest.skip("Supported")
    with pytest.raises(NotImplementedError, match="only of Parquet"):
        store.exec(select(QParty))


def test_embedded_types_sql_cannot_compare(db, store_kind):
    query = select(QParty).where(QParty.address.opened == dt.date(2020, 1, 1))
    if store_kind == "Parquet":
        assert names(db.exec(query)) == {"Belgo"}
    else:
        with pytest.raises(UnsupportedTypeError, match="address.opened"):
            db.exec(query)


def test_not_conditions():
    with pytest.raises(TypeError, match="Not a condition"):
        select(QUnregistered).where(QUnregistered.value == 5)
    with pytest.raises(TypeError, match="embedded QAddress"):
        select(QParty).where(QParty.address == BE.address)
    with pytest.raises(AttributeError, match="QParty has no field 'nme'"):
        _ = QTrade.party.nme  # pyright: ignore[reportAttributeAccessIssue]


def test_field_of_another_class(db):
    with pytest.raises(TypeError, match="QParty.country is not a field of QTrade"):
        db.exec(select(QTrade).where(QParty.country == "USA"))
    with pytest.raises(TypeError, match="embedded QAddress"):
        db.exec(select(QParty).where(col(QParty.address).is_(BE.address)))


def test_stores_not_supporting_queries():
    with pytest.raises(NotImplementedError, match="does not support queries"):
        PersistableStoreBase().exec(select(QParty))


def test_registered_again():
    QParty._registered()  # As is a type registered as an EntryPoint, whenever loaded
    assert isinstance(QParty.country, FieldColumn)
    assert US.country == "USA"


def test_classes_until_registered():
    # As their defaults, as dataclass takes the attributes of a class it is making to be
    assert QUnregistered.value == 5
    assert not hasattr(QUnregistered, "key")

    @dataclass(frozen=True, kw_only=True)
    class Sub(QParty):
        rating: int  # pyright: ignore[reportGeneralTypeIssues]
        extra: int = field(default=1)

    assert Sub.rating == 3  # pyright: ignore[reportAttributeAccessIssue]
    assert Sub(name="s", country="c").rating == 3

    class SubModel(QModel):
        extra: int = 1

    assert SubModel(key="k").value == 5
    assert not hasattr(SubModel, "value")
