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
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import CoreSchema

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
    active: bool = (
        True  # False once deleted. Records persisted before it was added are active
    )


class RecordFormat(Enum):
    Json = auto()
    Parquet = auto()


def _user() -> str:
    try:
        return getpass.getuser()
    except OSError:  # E.g. in a container, running as a uid with no name
        return "unknown"


class _LenientJsonSchema(GenerateJsonSchema):
    def handle_invalid_for_json_schema(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, schema: CoreSchema, error_info: str
    ) -> JsonSchemaValue:
        # A type with no JSON schema, e.g. an arbitrary class: noted, so that changing it changes the schema
        return {"description": error_info}


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

    def _check_types(self, types: set[type[Persistable]]):
        """Called by write before reading or writing any objects of types, e.g. to check that they can be"""

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

    @classmethod
    @cache
    def _json_schema(cls, typ: type[Persistable]) -> dict[str, Any]:
        """The JSON schema of typ's records"""
        return cls.__record_adapter(typ).json_schema(
            schema_generator=_LenientJsonSchema
        )

    def _validate(
        self, typ: type[Persistable], result: Any, resolver: ReferenceResolver
    ) -> ObjectRecord:
        """A result of _read, validated as a record of typ"""
        adapter = self.__record_adapter(typ)
        validate = (
            adapter.validate_json
            if self.__record_format is RecordFormat.Json
            else adapter.validate_python
        )
        return validate(result, context={RESOLVER: resolver})

    def read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime = dt.datetime.max,
        entry_time: dt.datetime = dt.datetime.max,
    ) -> dict[PersistableId, Persistable]:
        """The latest versions of the objects, as of the times, except those deleted"""
        records = self.__read(
            object_ids, effective_time, entry_time, ReferenceResolver()
        )
        return {i: r.data for i, r in records.items() if r.active}

    def __read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
        resolver: ReferenceResolver,
    ) -> dict[PersistableId, ObjectRecord]:
        by_type: dict[type[Persistable], set[PersistableId]] = {}
        for object_id in object_ids:
            by_type.setdefault(object_id.typ, set()).add(object_id)

        ret: dict[PersistableId, ObjectRecord] = {}
        unresolved = []  # Records with references, to validate again once those are loaded

        for typ, results in self._read(by_type, effective_time, entry_time).items():
            for result in results:
                misses = resolver.misses
                record = self._validate(typ, result, resolver)
                if resolver.misses > misses:
                    unresolved.append((typ, result))
                else:
                    ret[PersistableId.parse(record.object_id)] = self.__loaded(record)

        if unresolved:
            # Load the references in one read, as of the same times, except those this read has already loaded. They
            # resolve to their latest versions, even if deleted
            resolver.resolved.update({i: r.data for i, r in ret.items()})
            to_load = resolver.pending.difference(ret)
            loaded = (
                self.__read(to_load, effective_time, entry_time, ReferenceResolver())
                if to_load
                else {}
            )
            if missing := to_load.difference(loaded):
                raise RuntimeError(f"Failed to load references {missing!r}")

            resolver.resolved.update({i: r.data for i, r in loaded.items()})
            for typ, result in unresolved:
                record = self._validate(typ, result, resolver)
                ret[PersistableId.parse(record.object_id)] = self.__loaded(record)

        return ret

    @staticmethod
    def __loaded(record: ObjectRecord) -> ObjectRecord:
        record.data._set_object_entry_time(record.metadata.entry_time)
        return record

    def __current(
        self, object_ids: Iterable[PersistableId]
    ) -> dict[PersistableId, ObjectRecord]:
        # The latest records, including deleted ones. References are left as ids, which is all comparing records
        # needs: one read, rather than one per level
        return self.__read(
            object_ids,
            dt.datetime.max,
            dt.datetime.max,
            ReferenceResolver(resolve=False),
        )

    @staticmethod
    def __metadata(effective_time: dt.datetime) -> MetaData:
        entry_time = dt.datetime.now(dt.UTC)
        return MetaData(
            effective_time=effective_time
            if effective_time != dt.datetime.max
            else entry_time,
            entry_time=entry_time,
            host=socket.gethostname(),
            user=_user(),
        )

    def __dump(
        self,
        object_id: PersistableId,
        obj: Persistable,
        metadata: MetaData,
        context: dict[str, Any],
        active: bool = True,
    ) -> dict[str, Any]:
        record = self.record_type(type(obj))(
            object_id=str(object_id),
            metadata=metadata,
            data=obj,  # pyright: ignore[reportCallIssue]
            active=active,
        )
        parquet = self.__record_format is RecordFormat.Parquet
        return self.__record_adapter(type(obj)).dump_python(
            record, mode="python" if parquet else "json", context=context
        )

    def __conflict(
        self, object_id: PersistableId, obj: Persistable, stored: ObjectRecord
    ) -> str | None:
        # Why writing obj over the persisted version, stored, would be a conflict, if it would
        version, latest = obj.object_entry_time, stored.metadata.entry_time
        if version is None:
            # A deleted object may be restored without having been read
            if stored.active and self.__require_load_of_existing:
                return f"{object_id}, written at {latest}, was not read before being written"
        elif version != latest:
            return f"{object_id} was written at {latest}, since the version written at {version}"
        return None

    @staticmethod
    def __raise_conflicts(conflicts: list[str], force_overwrite: bool):
        if conflicts:
            if not force_overwrite:
                raise WriteConflictError("; ".join(conflicts))
            warnings.warn(f"Overwriting: {'; '.join(conflicts)}", stacklevel=3)

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

        metadata = self.__metadata(effective_time)

        # Serialise the objects and, if by_reference, those they reference, collected as each is serialised
        collector = ReferenceCollector()
        for obj in objects:
            collector.add(obj)

        context: dict[str, Any] = {STRUCT_LAYOUT: parquet}
        if by_reference:
            context[COLLECTOR] = collector

        objs: dict[PersistableId, Persistable] = {}
        records: dict[PersistableId, dict[str, Any]] = {}
        while new := collector.take():
            objs.update(new)
            for object_id, obj in new.items():
                collector.referrer = object_id
                records[object_id] = self.__dump(object_id, obj, metadata, context)

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

        # Before reading their current versions, which may fail if they cannot be
        self._check_types({type(obj) for obj in objs.values()})

        # Versions of the objects written or found unchanged
        versions: dict[PersistableId, dt.datetime] = {}
        conflicts: list[str] = []
        check_context = (
            {**context, COLLECTOR: ReferenceCollector()} if by_reference else context
        )

        for i, stored in self.__current(records).items():
            # Compare serialised data, in which references are ids: a change to a referenced object is not a change.
            # A deleted object is written even if unchanged, restoring it
            if (
                stored.active
                and self.__dump(i, stored.data, metadata, check_context)["data"]
                == records[i]["data"]
            ):
                del records[i]
                versions[i] = stored.metadata.entry_time
            elif conflict := self.__conflict(i, objs[i], stored):
                conflicts.append(conflict)

        self.__raise_conflicts(conflicts, force_overwrite)

        by_type: dict[type[Persistable], list[dict[str, Any]]] = {}
        for object_id, record in records.items():
            by_type.setdefault(object_id.typ, []).append(record)

        if by_type:
            self._write(by_type, metadata.entry_time)

        versions.update(dict.fromkeys(records, metadata.entry_time))
        for object_id, version in versions.items():
            objs[object_id]._set_object_entry_time(version)

    def delete(
        self,
        objects: Iterable[Persistable],
        effective_time: dt.datetime = dt.datetime.max,
        force_overwrite: bool = False,
    ):
        """Marks the objects deleted, as of effective_time, so that read no longer finds them until they are written again.

        Their persisted data is kept: reads as of earlier times find them. Objects not persisted, or already deleted, are
        ignored. Raises WriteConflictError, unless force_overwrite, as write does.
        """
        objs = {obj.object_id: obj for obj in objects}
        self._check_types({type(obj) for obj in objs.values()})
        current = {i: r for i, r in self.__current(objs).items() if r.active}
        self.__raise_conflicts(
            [c for i, r in current.items() if (c := self.__conflict(i, objs[i], r))],
            force_overwrite,
        )

        # The persisted data, not the objects', which may have been changed
        metadata = self.__metadata(effective_time)
        context = {STRUCT_LAYOUT: self.__record_format is RecordFormat.Parquet}
        by_type: dict[type[Persistable], list[dict[str, Any]]] = {}
        for object_id, stored in current.items():
            by_type.setdefault(object_id.typ, []).append(
                self.__dump(object_id, stored.data, metadata, context, active=False)
            )

        if by_type:
            self._write(by_type, metadata.entry_time)
            for object_id in current:
                objs[object_id]._set_object_entry_time(metadata.entry_time)

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
