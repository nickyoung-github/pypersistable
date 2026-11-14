import asyncio
import datetime as dt
import getpass
import socket
import warnings
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, make_dataclass
from enum import Enum, auto
from functools import cache
from graphlib import CycleError, TopologicalSorter
from typing import Any

from pydantic import ConfigDict, TypeAdapter

from .discriminated_union import STRUCT_LAYOUT
from .object_store import ObjectStore, WriteConflictError
from .persistable import Persistable
from .persistable_id import PersistableId
from .reference import COLLECTOR, RESOLVER, ReferenceCollector, ReferenceResolver


@dataclass
class MetaData:
    effective_time: dt.datetime
    entry_time: dt.datetime
    host: str
    user: str


@dataclass
class ObjectRecord[T]:  # There is a reason for this
    object_id: str
    metadata: MetaData
    data: T


class RecordFormat(Enum):
    Json = auto()
    Parquet = auto()


def _user() -> str:
    try:
        return getpass.getuser()
    except OSError:  # E.g. in a container, running as a uid with no name
        return "unknown"


class PersistableStore(ObjectStore):
    def __init__(
        self,
        record_format: RecordFormat,
        require_load_of_existing: bool = False,
    ):
        """require_load_of_existing: whether writing over a persisted object requires having read (or written) it"""
        self.__record_format = record_format
        self.__require_load_of_existing = require_load_of_existing

    @property
    def record_format(self) -> RecordFormat:
        return self.__record_format

    def _read(
        self,
        by_type: dict[type[Persistable], set[PersistableId]],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
    ) -> dict[type[Persistable], list[dict[str, Any]] | list[str]]:  # pyright: ignore[reportReturnType]
        """The latest records for the requested ids: raw JSON strings for RecordFormat.Json, else python objects"""

    def _write(
        self,
        by_type: dict[type[Persistable], list[dict[str, Any]]],
        entry_time: dt.datetime,
    ): ...

    @classmethod
    @cache
    def record_type(cls, typ: type[Persistable]) -> type[ObjectRecord]:
        return make_dataclass(
            f"{typ.__qualname__}Record",
            [("data", typ)],
            bases=(ObjectRecord,),
            # base64 in JSON, as bytes need not be valid UTF-8 (pydantic's default)
            namespace={
                "__pydantic_config__": ConfigDict(
                    ser_json_bytes="base64", val_json_bytes="base64"
                )
            },
        )

    @classmethod
    @cache
    def __record_adapter(cls, typ: type[Persistable]) -> TypeAdapter[ObjectRecord]:
        return TypeAdapter(cls.record_type(typ))

    def read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> dict[PersistableId, Persistable]:
        return self.__read(object_ids, effective_time, entry_time, ReferenceResolver())

    def __read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
        resolver: ReferenceResolver,
    ) -> dict[PersistableId, Persistable]:
        by_type: dict[type[Persistable], set[PersistableId]] = {}
        for object_id in object_ids:
            by_type.setdefault(object_id.typ, set()).add(object_id)

        ret: dict[PersistableId, Persistable] = {}
        context = {RESOLVER: resolver}
        unresolved = []  # Records with references, to validate again once those are loaded

        for typ, results in self._read(by_type, effective_time, entry_time).items():
            adapter = self.__record_adapter(typ)
            validate = (
                adapter.validate_json
                if self.__record_format is RecordFormat.Json
                else adapter.validate_python
            )

            for result in results:
                misses = resolver.misses
                record = validate(result, context=context)  # pyright: ignore[reportArgumentType]
                if resolver.misses > misses:
                    unresolved.append((validate, result))
                else:
                    ret[PersistableId.parse(record.object_id)] = self.__loaded(record)

        if unresolved:
            # Load the references in one read, as of the same times, except those this read has already loaded
            resolver.resolved.update(ret)
            to_load = resolver.pending.difference(ret)
            loaded = self.read(to_load, effective_time, entry_time) if to_load else {}
            if missing := to_load.difference(loaded):
                raise RuntimeError(f"Failed to load references {missing!r}")

            resolver.resolved.update(loaded)
            for validate, result in unresolved:
                record = validate(result, context=context)
                ret[PersistableId.parse(record.object_id)] = self.__loaded(record)

        return ret

    @staticmethod
    def __loaded(record: ObjectRecord) -> Persistable:
        record.data._set_object_entry_time(record.metadata.entry_time)
        return record.data

    def write(
        self,
        objects: Iterable[Persistable],
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
        by_reference: bool = True,
    ):
        parquet = self.__record_format is RecordFormat.Parquet
        if parquet and not by_reference:
            raise ValueError(
                "by_reference=False is not supported for RecordFormat.Parquet, which persists references as ids"
            )

        entry_time = dt.datetime.now(dt.UTC)
        metadata = MetaData(
            effective_time=effective_time
            if effective_time != dt.datetime.max
            else entry_time,
            entry_time=entry_time,
            host=socket.gethostname(),
            user=_user(),
        )

        # Serialise the objects and, if by_reference, those they reference, collected as each is serialised
        collector = ReferenceCollector()
        for obj in objects:
            collector.add(obj)

        context: dict[str, Any] = {STRUCT_LAYOUT: parquet}
        if by_reference:
            context[COLLECTOR] = collector

        def dump(object_id: PersistableId, obj: Persistable, context: dict[str, Any]):
            record = self.record_type(type(obj))(
                object_id=str(object_id),
                metadata=metadata,
                data=obj,  # pyright: ignore[reportCallIssue]
            )
            return self.__record_adapter(type(obj)).dump_python(
                record, mode="python" if parquet else "json", context=context
            )

        objs: dict[PersistableId, Persistable] = {}
        records: dict[PersistableId, dict[str, Any]] = {}
        while new := collector.take():
            objs.update(new)
            for object_id, obj in new.items():
                collector.referrer = object_id
                records[object_id] = dump(object_id, obj, context)

        if collector.conflicts:
            raise ValueError(
                f"Different objects with the same id: {', '.join(map(str, collector.conflicts))}"
            )

        try:
            TopologicalSorter(collector.graph).prepare()
        except CycleError as e:
            # Could not be read: each object must be constructed before those referencing it
            raise ValueError(
                f"Reference cycle: {' -> '.join(map(str, e.args[1]))}"
            ) from None

        # Versions of the objects written or found unchanged
        versions: dict[PersistableId, dt.datetime] = {}
        conflicts: list[str] = []
        # References are left as ids, which is all the comparison needs: one read, rather than one per level
        current = self.__read(
            records, dt.datetime.max, dt.datetime.max, ReferenceResolver(resolve=False)
        )
        check_context = (
            {**context, COLLECTOR: ReferenceCollector()} if by_reference else context
        )

        for i, stored in current.items():
            version, latest = objs[i].object_entry_time, stored.object_entry_time
            # Compare serialised data, in which references are ids: a change to a referenced object is not a change
            if dump(i, stored, check_context)["data"] == records[i]["data"]:
                del records[i]
                versions[i] = latest  # pyright: ignore[reportArgumentType]
            elif version is None:
                if self.__require_load_of_existing:
                    conflicts.append(
                        f"{i}, written at {latest}, was not read before being written"
                    )
            elif version != latest:
                conflicts.append(
                    f"{i} was written at {latest}, since the version written at {version}"
                )

        if conflicts:
            if not force_overwrite:
                raise WriteConflictError("; ".join(conflicts))
            warnings.warn(f"Overwriting: {'; '.join(conflicts)}", stacklevel=2)

        by_type: dict[type[Persistable], list[dict[str, Any]]] = {}
        for object_id, record in records.items():
            by_type.setdefault(object_id.typ, []).append(record)

        if by_type:
            self._write(by_type, entry_time)

        versions.update(dict.fromkeys(records, entry_time))
        for object_id, version in versions.items():
            objs[object_id]._set_object_entry_time(version)

    @staticmethod
    def read_all(
        *reads: Coroutine[Any, Any, Persistable],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> list[Persistable]:
        async def gather_reads():
            return await asyncio.gather(*reads)

        if loop is None:
            return asyncio.run(gather_reads())

        return loop.run_until_complete(gather_reads())
