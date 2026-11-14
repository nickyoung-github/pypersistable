import datetime as dt
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec import url_to_fs
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


class FSStore(PersistableStore):
    def __init__(
        self,
        url: str,
        record_format: RecordFormat = RecordFormat.Parquet,
        arrow_converter: ArrowConverter | None = None,
    ):
        """arrow_converter: for Parquet, defaults to persistable_arrow_converter()"""
        super().__init__(record_format)
        self.__arrow = arrow_converter or _ARROW_CONVERTER
        self.__fs, root = url_to_fs(url)
        self.__root = Path(root)
        self.__conn = duckdb.connect()
        self.__conn.register_filesystem(self.__fs)

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

        for typ, object_ids in by_type.items():
            files = f"{self.__directory(typ)!s}/**/*.{'parquet' if parquet else 'json'}"
            source = (
                f"read_parquet('{files}', union_by_name = true)"
                if parquet
                else f"read_json_objects('{files}', format = 'array')"
            )

            try:
                relation = self.__conn.sql(
                    f"""
                    SELECT {select}
                    FROM {source}
                    WHERE {object_id} IN (SELECT unnest(?))
                    AND {effective} <= ?
                    AND {entry} <= ?
                    QUALIFY row_number() OVER (PARTITION BY {object_id} ORDER BY {entry} DESC) = 1
                    """,
                    params=[[str(i) for i in object_ids], effective_time, entry_time],
                )
            except duckdb.IOException:
                if self.__fs.glob(files):
                    raise  # Not because nothing of this type has been persisted, e.g. a file could not be opened
                continue

            if parquet:
                # Cast to the schema written: duckdb returns e.g. durations as integers and timestamps in its own
                # time zone
                schema = self.__arrow.arrow_schema(self.record_type(typ))
                table = relation.to_arrow_table().select(schema.names).cast(schema)
                results[typ] = table.to_pylist(maps_as_pydicts="strict")
            else:
                results[typ] = [row for (row,) in relation.fetchall()]

        return results

    def _write(
        self,
        by_type: dict[type[Persistable], list[dict[str, Any]]],
        entry_time: dt.datetime,
    ):
        filename = entry_time.isoformat().replace(":", "~")

        for typ, records in by_type.items():
            directory = self.__directory(typ)
            self.__fs.makedirs(directory, exist_ok=True)
            path = directory / filename

            if self.record_format is RecordFormat.Parquet:
                schema = self.__arrow.arrow_schema(self.record_type(typ))
                table = pa.Table.from_pylist(
                    self.__arrow.arrow_value(records), schema=schema
                )
                with pq.ParquetWriter(
                    f"{path}.parquet", schema, filesystem=self.__fs
                ) as writer:
                    writer.write_table(table)
            else:
                self.__fs.write_bytes(f"{path}.json", dumps(records))
