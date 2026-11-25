import datetime as dt  # noqa: N999
from dataclasses import dataclass, field, replace
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from pypersistable import (
    Event,
    EventSourced,
    FSStore,
    Identifiable,
    IdField,
    Reference,
    TypeRegistry,
)

registry = TypeRegistry()


@dataclass(frozen=True, kw_only=True)
class DealState:
    """What a deal's events leave: not persisted"""

    notional: float


@dataclass(frozen=True)
class Deal(EventSourced[DealState]):
    """A label only: its state is that left by the events referencing it"""

    deal_id: str


registry.register("Deal", Deal)


@dataclass(frozen=True, kw_only=True)
class DealEvent(Event[DealState], Identifiable):
    """A base of events, so not registered"""

    deal: IdField[Reference[Deal]]
    event_id: IdField[UUID] = field(default_factory=uuid4)
    # The event it follows: not compared or shown, as that would be of all the events before it
    previous: "Reference[DealEvent] | None" = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True, kw_only=True)
class Initiated(DealEvent):
    notional: float

    def apply(self, state: DealState | None) -> DealState:
        return DealState(notional=self.notional)


registry.register("Initiated", Initiated)


@dataclass(frozen=True, kw_only=True)
class Upsized(DealEvent):
    by: float

    def apply(self, state: DealState | None) -> DealState:
        if state is None:
            raise ValueError("A deal is upsized only once initiated")
        return replace(state, notional=state.notional + self.by)


registry.register("Upsized", Upsized)


trade_date = dt.datetime(2026, 1, 5, tzinfo=dt.UTC)
upsize_date = dt.datetime(2026, 3, 2, tzinfo=dt.UTC)

root = (
    TemporaryDirectory()
)  # Replace with your store's URL, e.g. s3://my-bucket/ObjectStore
# Events are found by query, which FSStore supports for Parquet (its default), as does SQLStore
store = FSStore(f"file://{root.name}")

deal = Deal(deal_id="D1")
initiated = Initiated(deal=deal, notional=100)
# This will cause deal (because it is a reference) to be written too
initiated.write(store, effective_time=trade_date)
upsized = Upsized(deal=deal, previous=initiated, by=50)
upsized.write(store, effective_time=upsize_date)
known = upsized.object_entry_time
assert known is not None

print(f"Events: {deal.events(store)}")
print(f"State: {deal.state(store)}")
print(f"State as of the trade date: {deal.state(store, effective_time=trade_date)}\n")

# Each event follows the last
further = Upsized(deal=deal, previous=upsized, by=25)
further.write(store)
print(f"State once upsized again: {deal.state(store)}")

# An event is cancelled by deleting it: that following it then follows that before it
upsized.del_(store)
print(f"State with the first upsize cancelled: {deal.state(store)}")
print(f"State as it was known before: {deal.state(store, entry_time=known)}")
