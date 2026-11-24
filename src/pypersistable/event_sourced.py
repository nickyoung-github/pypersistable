"""Objects which are labels only, their state being that left by the events which reference them

    @dataclass(frozen=True)
    class Deal(EventSourced[DealState]):
        deal_id: str

    @dataclass(frozen=True, kw_only=True)
    class DealEvent(Event[DealState], Identifiable):
        deal: IdField[Reference[Deal]]
        previous: "Reference[DealEvent] | None"
        ...

    Deal(deal_id="D1").state(db, effective_time=last_month)

Events are found by query, so by a store supporting queries: install pypersistable[sql]
"""

import datetime as dt
from collections.abc import Collection
from functools import reduce
from operator import or_
from typing import Any

from .persistable import Persistable, Unique
from .persistable_id import PersistableId
from .persistable_store import PersistableStore
from .type_registry import TypeRegistry, UnregisteredTypeError


class Event[S](Persistable):
    """A change to the state, an S, of the EventSourced objects which its fields reference. Also an Identifiable or a
    Unique: its versions are those of the event, e.g. one corrected, or cancelled by deleting it.

    Its field which is a Reference to an Event is the event it follows: None, or no such field, if it is the first"""

    def apply(self, state: S | None) -> S:
        """The state this event leaves, given that before it: None if it is the first"""
        raise NotImplementedError


def _subclasses(cls: type) -> set[type]:
    return {s for c in cls.__subclasses__() for s in (c, *_subclasses(c))}


def _references(typ: type) -> dict[str, type]:
    """The fields of typ which are a Reference to one type, with it"""
    from .query import _hints, _target  # Lazily: it needs sqlalchemy

    return {
        name: target
        for name, hint in _hints(typ).items()
        for target, reference in [_target(hint)]
        if reference and target is not None
    }


def _before(event: Any, events: Collection[PersistableId]) -> PersistableId | None:
    """The id of the nearest of events which event follows: None if it is the first. Not, e.g., an event deleted"""
    while event is not None:
        fields = _references(type(event)).items()
        name = next((n for n, target in fields if issubclass(target, Event)), None)
        event = name and getattr(event, name)
        if event is not None and (object_id := event.object_id) in events:
            return object_id
    return None


class EventSourced[S](Unique):
    """A label, e.g. of a deal: its fields identify it, and its state, an S, is that left by its events"""

    def events(
        self,
        db: PersistableStore,
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> list[Event[S]]:
        """The events referencing this object, as of the times, as read: each after the event it follows.

        They are those of the registered subclasses of Event with a field which is a Reference to this object's type.
        Raises ValueError if two follow one event, or are first, e.g. if written concurrently
        """
        from .query import select

        events: dict[PersistableId, Event[S]] = {}
        for typ in _subclasses(Event):
            try:
                TypeRegistry().registered_name(typ)
            except UnregisteredTypeError:
                continue  # E.g. a base of events, or a class since redefined
            if conditions := [
                getattr(typ, name) == self
                for name, target in _references(typ).items()
                if isinstance(self, target)
            ]:
                statement = select(typ).where(reduce(or_, conditions))
                for event in db.exec(statement, effective_time, entry_time):
                    events[event.object_id] = event

        followers: dict[PersistableId | None, PersistableId] = {}
        for object_id, event in events.items():
            if (before := _before(event, events)) in followers:
                raise ValueError(
                    f"{followers[before]} and {object_id} both follow {before or 'no event'} of {self.object_id}"
                )
            followers[before] = object_id

        ordered: list[Event[S]] = []
        last = None
        while (last := followers.get(last)) is not None:
            ordered.append(events[last])
        return ordered

    def state(
        self,
        db: PersistableStore,
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> S | None:
        """The state left by this object's events, as of the times: None if there are none"""
        return reduce(
            lambda state, event: event.apply(state),
            self.events(db, effective_time, entry_time),
            None,
        )
