import datetime as dt
from functools import cache
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from anyschema import AnySchema
from fsspec import url_to_fs
from orjson import dumps

from ._anyschema import PIPELINE, dataclass_fields
from .persistable import Persistable
from .persistable_id import PersistableId
from .persistable_store import ObjectRecord, PersistableStore, RecordFormat
from .type_registry import TypeRegistry


class FSStore(PersistableStore):
    def __init__(self, url: str, record_format: RecordFormat = RecordFormat.Parquet):
        super().__init__(record_format)
        self.__fs, root = url_to_fs(url)
        self.__root = Path(root)
        self.__conn = duckdb.connect()
        self.__conn.register_filesystem(self.__fs)

    @classmethod
    @cache
    def __schema(cls, typ: type[ObjectRecord]) -> pa.Schema:
        schema = AnySchema(spec=dataclass_fields(typ), pipeline=PIPELINE)
        return schema.to_arrow()

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

            results[typ] = (
                relation.to_arrow_table().to_pylist()
                if parquet
                else [row for (row,) in relation.fetchall()]
            )

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
                schema = self.__schema(self.record_type(typ))
                table = pa.Table.from_pylist(records, schema=schema)
                with pq.ParquetWriter(
                    f"{path}.parquet", schema, filesystem=self.__fs
                ) as writer:
                    writer.write_table(table)
            else:
                self.__fs.write_bytes(f"{path}.json", dumps(records))
