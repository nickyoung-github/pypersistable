import datetime as dt
import getpass
import math
import socket
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, make_dataclass
from functools import cache
from graphlib import CycleError, TopologicalSorter
from typing import Any

from pydantic import ConfigDict
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import CoreSchema

from .persistable import Persistable
from .persistable_id import PersistableId
from .persistable_store import PersistableStore, WriteConflictError
from .reference import COLLECTOR, RESOLVER, ReferenceCollector, ReferenceResolver
from .typed_cache import type_adapter


@dataclass(kw_only=True)
class ObjectRecord:
    """
    A persisted version of an object. Stores derive the record type of each Persistable type, holding its data.
    Fields are prefixed with _, so that stores may hold them alongside the object's own fields
    """

    _object_id: str
    _effective_time: dt.datetime
    _entry_time: dt.datetime
    _host: str
    _user: str
    # False once deleted. Records persisted before it was added are active
    _active: bool = True


@dataclass(kw_only=True)
class _DataRecord(ObjectRecord):
    """A record holding its object whole: see PersistableStoreBase.record_type"""

    _data: Any


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


class PersistableStoreBase(PersistableStore):
    """Implements PersistableStore over the records which subclasses store, by implementing _read and _write"""

    def __init__(self, require_load_of_existing: bool = False):
        """require_load_of_existing: whether writing over a persisted object requires having read (or written) it"""
        self.__require_load_of_existing = require_load_of_existing

    def _read(
        self,
        by_type: dict[type[Persistable], set[PersistableId]],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
    ) -> dict[type[Persistable], list[Any]]:  # pyright: ignore[reportReturnType]
        """The latest records for the requested ids, as _validate accepts them"""

    def _write(
        self,
        batches: Sequence[tuple[type[Persistable], list[dict[str, Any]]]],
        entry_time: dt.datetime,
        expected: Mapping[PersistableId, dt.datetime | None],
    ):
        """Stores the records, as _serialise returns them, in batches of one type.

        Records reference only objects in their own or earlier batches, or stored already: a store which does not write
        atomically should write the batches in order, so that no reader finds a reference to an object not yet written.

        expected: the version (entry time) of each object which the write checked for conflicts, None if not persisted,
        or {} if forcing an overwrite. A store able to should raise WriteConflictError, writing nothing, if any has
        since changed, i.e. been written concurrently
        """

    @property
    def _references_by_value(self) -> bool:
        """Whether write supports by_reference=False, persisting referenced objects within those referencing them"""
        return True

    def _check_types(self, types: set[type[Persistable]]):
        """Called by write before reading or writing any objects of types, e.g. to check that they can be"""

    @classmethod
    @cache
    def record_type(cls, typ: type[Persistable]) -> type[_DataRecord]:
        return make_dataclass(
            f"{typ.__qualname__}Record",
            [("_data", typ)],
            bases=(_DataRecord,),
            kw_only=True,
            # base64 in JSON, as bytes need not be valid UTF-8 (pydantic's default)
            namespace={
                "__pydantic_config__": ConfigDict(
                    ser_json_bytes="base64", val_json_bytes="base64"
                )
            },
        )

    @classmethod
    @cache
    def _json_schema(cls, typ: type[Persistable]) -> dict[str, Any]:
        """The JSON schema of typ's records"""
        return type_adapter(cls.record_type(typ)).json_schema(
            schema_generator=_LenientJsonSchema
        )

    def _serialise(
        self, record: _DataRecord, context: dict[str, Any]
    ) -> dict[str, Any]:
        """A record, as _write stores it and as _read returns it"""
        return type_adapter(type(record)).dump_python(record, context=context)

    def _validate(
        self, typ: type[Persistable], result: Any, resolver: ReferenceResolver
    ) -> _DataRecord:
        """A result of _read, validated as a record of typ"""
        # By name too: records are dumped by field name, also for fields with aliases, which pydantic validates by
        return type_adapter(self.record_type(typ)).validate_python(
            result, context={RESOLVER: resolver}, by_name=True
        )

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
        return {i: r._data for i, r in records.items() if r._active}

    def __read(
        self,
        object_ids: Iterable[PersistableId],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
        resolver: ReferenceResolver,
    ) -> dict[PersistableId, _DataRecord]:
        ret: dict[PersistableId, _DataRecord] = {}
        # Records referencing objects not yet loaded, with the ids of those, to validate again once they are
        unresolved: dict[
            PersistableId, tuple[type[Persistable], Any, set[PersistableId]]
        ] = {}
        fetched: set[PersistableId] = set()
        to_fetch = requested = set(object_ids)
        # A level of references at a time, as of the same times. They resolve to their latest versions, even if deleted
        while to_fetch:
            fetched |= to_fetch
            by_type: dict[type[Persistable], set[PersistableId]] = {}
            for object_id in to_fetch:
                by_type.setdefault(object_id.typ, set()).add(object_id)

            to_fetch = set()
            for typ, results in self._read(by_type, effective_time, entry_time).items():
                for result in results:
                    resolver.pending = set()
                    record = self._validate(typ, result, resolver)
                    object_id = PersistableId.parse(record._object_id)
                    if resolver.pending:
                        unresolved[object_id] = (typ, result, resolver.pending)
                        to_fetch |= resolver.pending - fetched
                    else:
                        ret[object_id] = self.__loaded(record)
                        resolver.resolved[object_id] = record._data

        references = {i for *_, pending in unresolved.values() for i in pending}
        if missing := references - ret.keys() - unresolved.keys():
            raise RuntimeError(f"Failed to load references {missing!r}")

        # Those referenced first, so that each resolves
        graph = {i: pending for i, (*_, pending) in unresolved.items()}
        for object_id in TopologicalSorter(graph).static_order():
            if object_id in unresolved:
                typ, result, _ = unresolved[object_id]
                record = ret[object_id] = self.__loaded(
                    self._validate(typ, result, resolver)
                )
                resolver.resolved[object_id] = record._data

        return {i: r for i, r in ret.items() if i in requested}

    @staticmethod
    def __loaded(record: _DataRecord) -> _DataRecord:
        record._data._set_object_entry_time(record._entry_time)
        return record

    def __current(
        self, object_ids: Iterable[PersistableId]
    ) -> dict[PersistableId, _DataRecord]:
        # The latest records, including deleted ones. References are left as ids, which is all comparing records
        # needs: one read, rather than one per level
        return self.__read(
            object_ids,
            dt.datetime.max,
            dt.datetime.max,
            ReferenceResolver(resolve=False),
        )

    @staticmethod
    def __metadata(effective_time: dt.datetime) -> dict[str, Any]:
        """The ObjectRecord fields, other than _object_id and _active, of the records of a write"""
        entry_time = dt.datetime.now(dt.UTC)
        return {
            "_effective_time": effective_time
            if effective_time != dt.datetime.max
            else entry_time,
            "_entry_time": entry_time,
            "_host": socket.gethostname(),
            "_user": _user(),
        }

    def __dump(
        self,
        object_id: PersistableId,
        obj: Persistable,
        metadata: dict[str, Any],
        context: dict[str, Any],
        active: bool = True,
    ) -> dict[str, Any]:
        record = self.record_type(type(obj))(
            _object_id=str(object_id), _data=obj, _active=active, **metadata
        )
        return self._serialise(record, context)

    def __conflict(
        self,
        object_id: PersistableId,
        instances: Iterable[Persistable],
        stored: ObjectRecord,
    ) -> str | None:
        # Why writing the object over the persisted version, stored, would be a conflict, if it would. Its instances are
        # equal, so if any is of that version, none overwrites changes not seen
        latest = stored._entry_time
        versions = {v for obj in instances if (v := obj.object_entry_time) is not None}
        if not versions:
            # A deleted object may be restored without having been read
            if stored._active and self.__require_load_of_existing:
                return f"{object_id}, written at {latest}, was not read before being written"
        elif latest not in versions:
            return f"{object_id} was written at {latest}, since the version written at {max(versions)}"
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
        if not by_reference and not self._references_by_value:
            raise ValueError(
                f"by_reference=False is not supported by {self!r}, which persists references as ids"
            )

        metadata = self.__metadata(effective_time)
        entry_time: dt.datetime = metadata["_entry_time"]

        # Serialise the objects and, if by_reference, those they reference, collected as each is serialised
        collector = ReferenceCollector()
        for obj in objects:
            collector.add(obj)

        context: dict[str, Any] = {COLLECTOR: collector} if by_reference else {}

        records: dict[PersistableId, dict[str, Any]] = {}
        while new := collector.take():
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
        instances = collector.instances
        self._check_types({type(objs[0]) for objs in instances.values()})

        # Versions of the objects written or found unchanged
        versions: dict[PersistableId, dt.datetime] = {}
        conflicts: list[str] = []
        check_context = {COLLECTOR: ReferenceCollector()} if by_reference else {}

        current = self.__current(records)
        for i, stored in current.items():
            # Compare serialised data, in which references are ids: a change to a referenced object is not a change.
            # A deleted object is written even if unchanged, restoring it
            if (
                stored._active
                # Else, e.g. if referenced by an object of an earlier effective time, not found by reads as of it
                and stored._effective_time <= metadata["_effective_time"]
                and self.__dump(i, stored._data, metadata, check_context)["_data"]
                == records[i]["_data"]
            ):
                del records[i]
                versions[i] = stored._entry_time
            elif conflict := self.__conflict(i, instances[i], stored):
                conflicts.append(conflict)

        self.__raise_conflicts(conflicts, force_overwrite)

        if records:
            expected = {
                i: s._entry_time if (s := current.get(i)) else None for i in records
            }
            self._write(
                _batches(records, collector.graph),
                entry_time,
                {} if force_overwrite else expected,
            )

        versions.update(dict.fromkeys(records, entry_time))
        for object_id, version in versions.items():
            for obj in instances[object_id]:
                obj._set_object_entry_time(version)

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
        instances: dict[PersistableId, list[Persistable]] = {}
        for obj in objects:
            instances.setdefault(obj.object_id, []).append(obj)
        self._check_types({type(objs[0]) for objs in instances.values()})
        current = {i: r for i, r in self.__current(instances).items() if r._active}
        self.__raise_conflicts(
            [
                c
                for i, r in current.items()
                if (c := self.__conflict(i, instances[i], r))
            ],
            force_overwrite,
        )

        # The persisted data, not the objects', which may have been changed
        metadata = self.__metadata(effective_time)
        entry_time: dt.datetime = metadata["_entry_time"]
        by_type: dict[type[Persistable], list[dict[str, Any]]] = {}
        for object_id, stored in current.items():
            by_type.setdefault(object_id.typ, []).append(
                self.__dump(object_id, stored._data, metadata, {}, active=False)
            )

        if by_type:
            expected = {i: r._entry_time for i, r in current.items()}
            self._write(
                list(by_type.items()),
                entry_time,
                {} if force_overwrite else expected,
            )
            for object_id in current:
                for obj in instances[object_id]:
                    obj._set_object_entry_time(entry_time)


def _batches(
    records: Mapping[PersistableId, dict[str, Any]],
    graph: Mapping[PersistableId, set[PersistableId]],
) -> list[tuple[type[Persistable], list[dict[str, Any]]]]:
    """The records, in few batches of one type, each after the batches of the objects it references"""
    batches: list[tuple[type[Persistable], list[dict[str, Any]]]] = []
    positions: dict[PersistableId, int] = {}
    # Referenced objects first
    for i in TopologicalSorter({i: graph.get(i, ()) for i in records}).static_order():
        if i not in records:
            continue  # Unchanged, so not written
        # In or after the batches of the objects it references: after, if of another type
        after = max(
            (
                positions[r] + (r.typ is not i.typ)
                for r in graph.get(i, ())
                if r in positions
            ),
            default=0,
        )
        positions[i] = next(
            (p for p in range(after, len(batches)) if batches[p][0] is i.typ),
            len(batches),
        )
        if positions[i] == len(batches):
            batches.append((i.typ, []))
        batches[positions[i]][1].append(records[i])
    return batches


def _check_finite(record: Mapping[str, Any]):
    """Raises ValueError if a record, as python mode serialisation returns it, holds NaN or an infinity: JSON has none,
    so pydantic would write null"""

    def non_finite(value: Any) -> bool:
        if isinstance(value, dict):
            return any(map(non_finite, value.values()))
        if isinstance(value, list | tuple | set | frozenset):
            return any(map(non_finite, value))
        return isinstance(value, float) and not math.isfinite(value)

    if non_finite(record["_data"]):
        raise ValueError(
            f"{record['_object_id']} holds NaN or an infinity, which JSON cannot represent"
        )
