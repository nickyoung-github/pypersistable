"""Event-sourced objects, of every store supporting queries: see conftest.STORE_KINDS"""

import datetime as dt
import warnings
from dataclasses import dataclass, field, replace
from uuid import UUID, uuid4

import pytest
from models import registry
from pydantic import BaseModel, Field
from pydantic.warnings import GenericBeforeBaseModelWarning

from pypersistable import (
    Event,
    EventSourced,
    Identifiable,
    IdField,
    PersistableStoreBase,
    Reference,
    Unique,
)


@dataclass(frozen=True, kw_only=True)
class DealState:
    notional: float
    status: str = "live"


@dataclass(frozen=True)
class EDeal(EventSourced[DealState]):
    deal_id: str


@dataclass(frozen=True, kw_only=True)
class EDealEvent(Event[DealState], Identifiable):
    """Not registered: a base of events"""

    deal: IdField[Reference[EDeal]]
    event_id: IdField[UUID] = field(default_factory=uuid4)
    previous: "Reference[EDealEvent] | None" = None


@dataclass(frozen=True, kw_only=True)
class EInitiated(EDealEvent):
    notional: float

    def apply(self, state: DealState | None) -> DealState:
        assert state is None
        return DealState(notional=self.notional)


@dataclass(frozen=True, kw_only=True)
class EUpsized(EDealEvent):
    by: float

    def apply(self, state: DealState | None) -> DealState:
        assert state is not None
        return replace(state, notional=state.notional + self.by)


@dataclass(frozen=True, kw_only=True)
class EExpired(Event[DealState], Unique):
    """A Unique: there is only one such event of a deal"""

    deal: Reference[EDeal]
    previous: Reference[EDealEvent]

    def apply(self, state: DealState | None) -> DealState:
        assert state is not None
        return replace(state, status="expired")


@dataclass(frozen=True, kw_only=True)
class ENovated(EDealEvent):
    """Of two deals: that novated, and that replacing it"""

    to: Reference[EDeal]

    def apply(self, state: DealState | None) -> DealState:
        return DealState(notional=0, status="novated")


@dataclass(frozen=True, kw_only=True)
class EBook(Identifiable):
    """References deals, but is not an event of them"""

    name: IdField[str]
    deals: tuple[Reference[EDeal], ...] = ()
    lead: Reference[EDeal] | None = None


# pydantic warns of generic bases before BaseModel, which Persistable must be
with warnings.catch_warnings():
    warnings.simplefilter("ignore", GenericBeforeBaseModelWarning)

    class MAccount(EventSourced[int], BaseModel, frozen=True):
        number: str

    class MDeposited(Event[int], Identifiable, BaseModel):
        account: IdField[Reference[MAccount]]
        event_id: IdField[UUID] = Field(default_factory=uuid4)
        previous: "Reference[MDeposited] | None" = None
        amount: int

        def apply(self, state: int | None) -> int:
            return (state or 0) + self.amount


for _typ in (EDeal, EInitiated, EUpsized, EExpired, ENovated, EBook, MAccount, MDeposited):  # fmt: skip
    registry.register(_typ.__name__, _typ)

T1, T2, T3 = (dt.datetime(2026, m, 1, tzinfo=dt.UTC) for m in (1, 2, 3))
D1, D2 = EDeal(deal_id="D1"), EDeal(deal_id="D2")


@pytest.fixture
def db(store_kind, store) -> PersistableStoreBase:
    if store_kind == "Json":
        pytest.skip("FSStore queries Parquet only")
    return store


def test_state_is_that_left_by_events(db):
    assert D1.events(db) == []
    assert D1.state(db) is None

    initiated = EInitiated(deal=D1, notional=100)
    initiated.write(db)
    upsized = EUpsized(deal=D1, previous=initiated, by=50)
    upsized.write(db)
    expired = EExpired(deal=D1, previous=upsized)
    expired.write(db)
    # Of another deal
    EInitiated(deal=D2, notional=7).write(db)
    # Not an event
    EBook(name="b", deals=(D1,), lead=D1).write(db)

    assert D1.events(db) == [initiated, upsized, expired]
    assert D1.state(db) == DealState(notional=150, status="expired")
    assert D2.state(db) == DealState(notional=7)
    # The label was written with its events
    assert EDeal.read(db, deal_id="D1") == D1


def test_events_are_ordered_by_those_they_follow_not_by_time(db):
    initiated = EInitiated(deal=D1, notional=100)
    upsized = EUpsized(deal=D1, previous=initiated, by=50)
    expired = EExpired(deal=D1, previous=upsized)
    # With those it follows: all of one effective and entry time
    expired.write(db, effective_time=T1)
    assert D1.events(db) == [initiated, upsized, expired]

    # Corrected, so now the last written
    corrected = replace(initiated, notional=200)
    corrected.write(db, effective_time=T1)
    assert D1.events(db)[0] == corrected
    assert D1.state(db) == DealState(notional=250, status="expired")


def test_state_as_of_effective_time(db):
    initiated = EInitiated(deal=D1, notional=100)
    initiated.write(db, effective_time=T1)
    EUpsized(deal=D1, previous=initiated, by=50).write(db, effective_time=T2)

    assert D1.state(db) == DealState(notional=150)
    assert D1.state(db, effective_time=T1) == DealState(notional=100)
    assert D1.state(db, effective_time=T1 - dt.timedelta(days=1)) is None


def test_event_is_effective_no_later_than_those_following_it(db):
    initiated = EInitiated(deal=D1, notional=100)
    initiated.write(db, effective_time=T2)
    # Backdated: the store writes the event it references again, as of its effective time
    EUpsized(deal=D1, previous=initiated, by=50).write(db, effective_time=T1)

    assert D1.state(db, effective_time=T1) == DealState(notional=150)


def test_state_as_of_entry_time(db):
    initiated = EInitiated(deal=D1, notional=100)
    initiated.write(db)
    known = initiated.object_entry_time
    assert known is not None
    EUpsized(deal=D1, previous=initiated, by=50).write(db)
    replace(initiated, notional=200).write(db)

    assert D1.state(db) == DealState(notional=250)
    assert D1.state(db, entry_time=known) == DealState(notional=100)


def test_event_cancelled_by_deleting_it(db):
    initiated = EInitiated(deal=D1, notional=100)
    cancelled = EUpsized(deal=D1, previous=initiated, by=50)
    upsized = EUpsized(deal=D1, previous=cancelled, by=25)
    upsized.write(db)
    known = upsized.object_entry_time
    assert known is not None

    cancelled.del_(db)

    # That following it now follows that before it
    assert D1.events(db) == [initiated, upsized]
    assert D1.state(db) == DealState(notional=125)
    assert D1.state(db, entry_time=known) == DealState(notional=175)


def test_event_inserted_by_writing_that_following_it_again(db):
    initiated = EInitiated(deal=D1, notional=100)
    expired = EExpired(deal=D1, previous=initiated)
    expired.write(db)

    upsized = EUpsized(deal=D1, previous=initiated, by=50)
    upsized.write(db)
    with pytest.raises(ValueError, match="both follow EInitiated/"):
        D1.events(db)

    # A Unique, so a new event
    expired.del_(db)
    after = replace(expired, previous=upsized)
    after.write(db)
    assert D1.events(db) == [initiated, upsized, after]


def test_two_first_events(db):
    EInitiated(deal=D1, notional=100).write(db)
    EInitiated(deal=D1, notional=200).write(db)

    with pytest.raises(ValueError, match="both follow no event of EDeal/"):
        D1.state(db)


def test_event_of_two_objects(db):
    initiated = EInitiated(deal=D1, notional=100)
    novated = ENovated(deal=D1, previous=initiated, to=D2)
    upsized = EUpsized(deal=D2, previous=novated, by=5)
    upsized.write(db)

    assert D1.events(db) == [initiated, novated]
    # Its first: the event it follows is not one of it
    assert D2.events(db) == [novated, upsized]


def test_models(db):
    account = MAccount(number="1")
    first = MDeposited(account=account, amount=5)
    MDeposited(account=account, previous=first, amount=10).write(db)

    assert [e.amount for e in account.events(db)] == [5, 10]  # pyright: ignore[reportAttributeAccessIssue]
    assert account.state(db) == 15
    assert MAccount(number="2").state(db) is None


def test_redefined_event_class(db):
    def define(notional: float) -> type[EDealEvent]:
        @dataclass(frozen=True, kw_only=True)
        class Redefined(EDealEvent):
            def apply(self, state: DealState | None) -> DealState:
                return DealState(notional=notional)

        return Redefined

    old, new = define(1), define(2)
    registry.register("ERedefined", old, replace=True)
    registry.register("ERedefined", new, replace=True)
    new(deal=D1).write(db)

    # Of the class registered, not that it replaced
    assert D1.state(db) == DealState(notional=2)


def test_event_must_apply(db):
    @dataclass(frozen=True, kw_only=True)
    class Inapplicable(EDealEvent): ...

    registry.register("EInapplicable", Inapplicable, replace=True)
    Inapplicable(deal=D1).write(db)

    with pytest.raises(NotImplementedError):
        D1.state(db)


def test_json_not_supported(store_kind, store):
    if store_kind != "Json":
        pytest.skip("Supported")
    EInitiated(deal=D1, notional=100).write(store)

    with pytest.raises(NotImplementedError, match="does not support queries"):
        D1.state(store)
