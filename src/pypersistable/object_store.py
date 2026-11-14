import asyncio
import datetime as dt
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from .persistable_id import PersistableId


class UnknownObjectError(KeyError):
    def __init__(self, typ: type, kwargs: dict[str, Any], object_id: PersistableId):
        self.typ = typ
        self.kwargs = kwargs
        self.object_id = object_id
        super().__init__(f"Unknown object {object_id}")

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"No object found for {self.typ.__qualname__}({args}) [{self.object_id}]"


class ObjectStore:
    def read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> dict[PersistableId, Any]: ...

    def write(
        self,
        objects: Iterable[Any],
        effective_time: dt.datetime = dt.datetime.max,
        check: bool = True,
        by_reference: bool = True,
    ): ...


@dataclass(frozen=True)
class _ReadKey:
    db_id: int
    effective_time: dt.datetime
    entry_time: dt.datetime


class ReadBatch:
    __current: ClassVar[ContextVar[Self | None]] = ContextVar(
        "ReadBatch.current", default=None
    )

    def __init__(self):
        self.__pending: dict[_ReadKey, tuple[ObjectStore, set[PersistableId]]] = {}
        self.__tasks: dict[_ReadKey, asyncio.Future[dict[PersistableId, Any]]] | None = None

    @classmethod
    def current(cls) -> Self:
        if (current := cls.__current.get()) is None or current.dispatched:
            current = cls()
            cls.__current.set(current)
        return current

    @property
    def dispatched(self) -> bool:
        return self.__tasks is not None

    def dispatch(self):
        if self.dispatched:
            return

        pending, self.__pending = self.__pending, {}
        self.__tasks = {
            key: asyncio.ensure_future(
                asyncio.to_thread(db.read, oids, key.effective_time, key.entry_time)
            )
            for key, (db, oids) in pending.items()
        }

    def enqueue(
        self,
        db: ObjectStore,
        typ: type,
        object_id: PersistableId,
        kwargs: dict[str, Any],
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ):
        if self.dispatched:
            raise RuntimeError("Cannot enqueue after dispatch")

        key = _ReadKey(id(db), effective_time, entry_time)
        self.__pending.setdefault(key, (db, set()))[1].add(object_id)
        return self.__result(key, typ, object_id, dict(kwargs))

    async def __result(
        self, key: _ReadKey, typ: type, object_id: PersistableId, kwargs: dict[str, Any]
    ) -> Any:
        self.dispatch()
        assert self.__tasks is not None

        results = await self.__tasks[key]
        if object_id not in results:
            raise UnknownObjectError(typ, kwargs, object_id)

        return results[object_id]
