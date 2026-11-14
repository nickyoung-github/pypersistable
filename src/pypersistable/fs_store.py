import datetime as dt
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import NoneType, UnionType
from typing import Annotated, Any, Union, get_args, get_origin
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from orjson import dumps

from .arrow import ArrowConverter, UnsupportedTypeError
from .discriminated_union import TAG_FIELD, DiscriminatedUnionInfo
from .persistable import Persistable
from .persistable_id import PersistableId
from .persistable_store import PersistableStore, RecordFormat
from .reference import _Reference
from .type_registry import TypeRegistry


def persistable_arrow_converter() -> ArrowConverter:
    """An ArrowConverter for Persistables, handling Reference and DiscriminatedUnion. Add handlers for your own types"""

    def union(_: ArrowConverter, hint: Any, __: tuple[Any, ...]) -> None:
        # Only T | None is supported (by ArrowConverter): point to the alternatives
        if (
            get_origin(hint) in (Union, UnionType)
            and len(members := set(get_args(hint)) - {NoneType}) > 1
        ):
            persistable = all(
                isinstance(m, type) and issubclass(m, Persistable) for m in members
            )
            raise UnsupportedTypeError(
                f"Unsupported union {hint}: only T | None is supported. "
                f"Use {'Reference[...] or ' if persistable else ''}DiscriminatedUnion[...] instead"
            )

    def reference(
        _: ArrowConverter, __: Any, metadata: tuple[Any, ...]
    ) -> pa.DataType | None:
        # Persisted as its object id
        return pa.string() if any(isinstance(m, _Reference) for m in metadata) else None

    def discriminated_union(
        converter: ArrowConverter, _: Any, metadata: tuple[Any, ...]
    ) -> pa.DataType | None:
        # A struct of the tag and a nullable field per member: see discriminated_union.STRUCT_LAYOUT
        for info in (m for m in metadata if isinstance(m, DiscriminatedUnionInfo)):
            members = ((tag, converter.arrow_type(t)) for tag, t in info.members)
            return pa.struct([(TAG_FIELD, pa.string()), *members])
        return None

    converter = ArrowConverter()
    converter.add_handler(union)  # Added first, so checked after the others
    converter.add_handler(reference)
    converter.add_handler(discriminated_union)
    return converter


_ARROW_CONVERTER = persistable_arrow_converter()


def _drop_absent(value: Any, arrow_type: pa.DataType, file_type: pa.DataType) -> Any:
    # Drops the struct fields of arrow_type absent from file_type, i.e. those added since the file was written, which
    # union_by_name and casting fill with nulls: pydantic then applies their defaults
    if value is None or arrow_type == file_type:
        return value
    if pa.types.is_struct(arrow_type):
        file_fields = {f.name: f.type for f in file_type}
        return {
            name: _drop_absent(v, arrow_type.field(name).type, file_fields[name])
            for name, v in value.items()
            if name in file_fields
        }
    if pa.types.is_map(arrow_type):
        return {
            k: _drop_absent(v, arrow_type.item_type, file_type.item_type)
            for k, v in value.items()
        }
    if pa.types.is_list(arrow_type):
        return [
            _drop_absent(v, arrow_type.value_type, file_type.value_type) for v in value
        ]
    return value


def _differences(
    arrow_type: pa.DataType, file_type: pa.DataType, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], str | None]]:
    # (path, type change) of the fields removed (None), or whose types have changed. Added fields are compatible: their
    # defaults apply on reading
    if arrow_type == file_type:
        return
    if pa.types.is_struct(arrow_type) and pa.types.is_struct(file_type):
        fields = {f.name: f.type for f in arrow_type}
        for f in file_type:
            if f.name in fields:
                yield from _differences(fields[f.name], f.type, (*path, f.name))
            else:
                yield (*path, f.name), None
    elif pa.types.is_map(arrow_type) and pa.types.is_map(file_type):
        yield from _differences(arrow_type.key_type, file_type.key_type, (*path, "key"))
        yield from _differences(
            arrow_type.item_type, file_type.item_type, (*path, "value")
        )
    elif pa.types.is_list(arrow_type) and pa.types.is_list(file_type):
        yield from _differences(
            arrow_type.value_type, file_type.value_type, (*path, "item")
        )
    else:
        yield path, f"{arrow_type}, but {file_type} when persisted"


def _unwrap(hint: Any) -> tuple[Any, tuple[Any, ...]]:
    # The type and Annotated metadata of a hint, which, if a union, is persisted, so is T | None
    metadata = ()
    while True:
        if get_origin(hint) is Annotated:
            hint, *extra = get_args(hint)
            metadata += tuple(extra)
        elif get_origin(hint) in (Union, UnionType):
            hint = next(a for a in get_args(hint) if a is not NoneType)
        else:
            return hint, metadata


class IncompatibleSchemaError(TypeError):
    """A type's schema is incompatible with that of its persisted objects: a field was removed or its type changed"""


class FSStore(PersistableStore):
    def __init__(
        self,
        url: str,
        record_format: RecordFormat = RecordFormat.Parquet,
        arrow_converter: ArrowConverter | None = None,
        require_load_of_existing: bool = False,
    ):
        """arrow_converter: for Parquet, defaults to persistable_arrow_converter()"""
        super().__init__(record_format, require_load_of_existing)
        self.__arrow = arrow_converter or _ARROW_CONVERTER
        self.__fs, root = url_to_fs(url)
        self.__root = Path(root)
        self.__conn = duckdb.connect()
        self.__conn.register_filesystem(self.__fs)
        # Schemas of Parquet files by path, which are never rewritten
        self.__file_types: dict[str, pa.StructType] = {}
        self.__checked_files: set[str] = set()  # For ignored fields, when read

    def __duckdb_path(self, path: str) -> str:
        # duckdb uses the registered filesystem for paths with its protocol, and its own for local paths (faster)
        return (
            path
            if isinstance(self.__fs, LocalFileSystem)
            else self.__fs.unstrip_protocol(path)
        )

    def __differences(
        self, typ: type[Persistable], file: str
    ) -> tuple[list[str], list[str]]:
        """The incompatible and the ignored (removed, but allowed as extra) fields of a Parquet file of typ"""
        record_type = self.record_type(typ)
        incompatible, ignored = [], []
        arrow_type = pa.struct(self.__arrow.arrow_schema(record_type))
        for path, change in _differences(arrow_type, self.__file_type(file)):
            name = ".".join(path)
            if change is not None:
                incompatible.append(f"{name} is {change}")
            elif self.__allows_extra(record_type, path[:-1]):
                ignored.append(name)
            else:
                incompatible.append(f"{name} has been removed")
        return incompatible, ignored

    def __allows_extra(self, hint: Any, path: tuple[str, ...]) -> bool:
        # Whether the struct-like type at path (as from _differences) within hint allows extra fields
        for segment in path:
            hint, metadata = _unwrap(hint)
            if info := next(
                (m for m in metadata if isinstance(m, DiscriminatedUnionInfo)), None
            ):
                hint = dict(info.members)[segment]
            elif get_origin(hint) in (dict, Mapping):
                hint = get_args(hint)[segment == "value"]
            elif get_origin(hint) is not None:
                hint = get_args(hint)[0]  # A collection
            else:
                hint = self.__arrow.struct_fields(hint)[segment]  # pyright: ignore[reportOptionalSubscript]

        hint, metadata = _unwrap(hint)
        if any(isinstance(m, DiscriminatedUnionInfo) for m in metadata):
            return False  # A removed member's values could not be validated
        config = getattr(hint, "model_config", None) or getattr(
            hint, "__pydantic_config__", {}
        )
        return config.get("extra") != "forbid"

    def __file_type(self, path: str) -> pa.StructType:
        if (file_type := self.__file_types.get(path)) is None:
            file_type = self.__file_types[path] = pa.struct(
                pq.read_schema(path, filesystem=self.__fs)
            )
        return file_type

    def __directory(self, typ: type[Persistable]) -> Path:
        # Store root / TypeRegistry root / registered name
        registry = TypeRegistry[Persistable]()
        return self.__root / registry.root(typ) / registry.registered_name(typ)

    def _read(
        self,
        by_type: dict[type[Persistable], set[PersistableId]],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
    ) -> dict[type[Persistable], list[dict[str, Any]] | list[str]]:
        results = {}
        parquet = self.record_format is RecordFormat.Parquet

        if parquet:
            select, object_id, effective, entry = (
                "*",
                "object_id",
                "metadata.effective_time",
                "metadata.entry_time",
            )
        else:
            # Raw JSON: pydantic validates it, rather than duckdb inferring (and merging) a schema across records
            select = "json"
            object_id = "json->>'$.object_id'"
            effective, entry = (
                f"(json->>'$.metadata.{f}')::TIMESTAMPTZ"
                for f in ("effective_time", "entry_time")
            )

        # A cursor per read, as reads may run concurrently (see ReadBatch), which a connection does not support
        with self.__conn.cursor() as conn:
            for typ, object_ids in by_type.items():
                files = (
                    f"{self.__directory(typ)!s}/**/*.{'parquet' if parquet else 'json'}"
                )
                source = (
                    "read_parquet(?, union_by_name = true, filename = true)"
                    if parquet
                    else "read_json_objects(?, format = 'array')"
                )

                try:
                    relation = conn.sql(
                        f"""
                        SELECT {select}
                        FROM {source}
                        WHERE {object_id} IN (SELECT unnest(?))
                        AND {effective} <= ?
                        AND {entry} <= ?
                        QUALIFY row_number() OVER (PARTITION BY {object_id} ORDER BY {entry} DESC) = 1
                        """,
                        params=[
                            self.__duckdb_path(files),
                            [str(i) for i in object_ids],
                            effective_time,
                            entry_time,
                        ],
                    )
                except duckdb.IOException:
                    if self.__fs.glob(files):
                        raise  # Not because nothing of this type has been persisted, e.g. a file could not be opened
                    continue

                if parquet:
                    # Cast to the schema written: duckdb returns e.g. durations as integers and timestamps in its own
                    # time zone
                    schema = self.__arrow.arrow_schema(self.record_type(typ))
                    table = relation.to_arrow_table()
                    records = (
                        table.select(schema.names)
                        .cast(schema)
                        .to_pylist(maps_as_pydicts="strict")
                    )
                    filenames = table["filename"].to_pylist()
                    record_type = pa.struct(schema)
                    results[typ] = [
                        _drop_absent(r, record_type, self.__file_type(f))
                        for r, f in zip(records, filenames, strict=True)
                    ]

                    for f in set(filenames) - self.__checked_files:
                        self.__checked_files.add(f)
                        if ignored := self.__differences(typ, f)[1]:
                            warnings.warn(
                                f"{f}: ignoring {', '.join(ignored)}, removed from {typ.__qualname__}",
                                stacklevel=2,
                            )
                else:
                    results[typ] = [row for (row,) in relation.fetchall()]

        return results

    def _write(
        self,
        by_type: dict[type[Persistable], list[dict[str, Any]]],
        entry_time: dt.datetime,
    ):
        # Unique, as concurrent writers may share an entry time. Not ":", which some file systems disallow
        filename = f"{entry_time.isoformat().replace(':', '~')}-{uuid4().hex[:8]}"
        parquet = self.record_format is RecordFormat.Parquet

        if parquet:
            # Before writing anything
            for typ in by_type:
                files = self.__fs.glob(f"{self.__directory(typ)}/**/*.parquet")
                if problems := {
                    p for f in files for p in self.__differences(typ, f)[0]
                }:
                    raise IncompatibleSchemaError(
                        f"{typ.__qualname__}: {'; '.join(sorted(problems))}"
                    )

        for typ, records in by_type.items():
            directory = self.__directory(typ)
            self.__fs.makedirs(directory, exist_ok=True)
            path = directory / filename

            if parquet:
                schema = self.__arrow.arrow_schema(self.record_type(typ))
                table = pa.Table.from_pylist(
                    self.__arrow.arrow_value(records, pa.list_(pa.struct(schema))),
                    schema=schema,
                )
                with pq.ParquetWriter(
                    f"{path}.parquet", schema, filesystem=self.__fs
                ) as writer:
                    writer.write_table(table)
            else:
                self.__fs.write_bytes(f"{path}.json", dumps(records))
