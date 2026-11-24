import asyncio
import datetime as dt
from collections.abc import Coroutine, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from .persistable_id import PersistableId

if TYPE_CHECKING:
    from .persistable import Persistable
    from .query import Select


class WriteConflictError(Exception):
    """Writing an object would overwrite a version of it which the writer has not seen"""


class UnknownObjectError(KeyError):
    def __init__(self, typ: type, kwargs: dict[str, Any], object_id: PersistableId):
        self.typ = typ
        self.kwargs = kwargs
        self.object_id = object_id
        super().__init__(f"Unknown object {object_id}")

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"No object found for {self.typ.__qualname__}({args}) [{self.object_id}]"


class PersistableStore:
    def read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> "dict[PersistableId, Persistable]": ...

    def write(
        self,
        objects: "Iterable[Persistable]",
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
        by_reference: bool = True,
    ): ...

    def delete(
        self,
        objects: "Iterable[Persistable]",
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
    ): ...

    def exec[T: "Persistable"](
        self,
        statement: "Select[T]",
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> list[T]: ...

    @staticmethod
    def read_all(
        *reads: "Coroutine[Any, Any, Persistable]",
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> "list[Persistable]":
        async def gather_reads():
            return await asyncio.gather(*reads)

        if loop is None:
            return asyncio.run(gather_reads())

        return loop.run_until_complete(gather_reads())


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
        self.__pending: dict[_ReadKey, tuple[PersistableStore, set[PersistableId]]] = {}
        self.__tasks: (
            dict[_ReadKey, asyncio.Future[dict[PersistableId, Any]]] | None
        ) = None

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
        db: PersistableStore,
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
