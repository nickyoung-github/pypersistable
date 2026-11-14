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

from ._anyschema import PIPELINE
from .persistable_id import PersistableId
from .persistable_store import ObjectRecord, PersistableStore, RecordFormat


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
        schema = AnySchema(spec=typ, pipeline=PIPELINE)
        return schema.to_arrow()

    def _read(
        self,
        by_directory: dict[Path, list[PersistableId]],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
    ) -> dict[Path, list[dict[str, Any]] | list[str]]:
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

        for directory, object_ids in by_directory.items():
            files = f"{self.__root.joinpath(directory)!s}/**/*"
            source = (
                f"read_parquet('{files}.parquet', union_by_name = true)"
                if parquet
                else f"read_json_objects('{files}.json', format = 'array')"
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
                # This will happen if no objects of this type have been saved
                continue

            results[directory] = (
                relation.to_arrow_table().to_pylist()
                if parquet
                else [row for (row,) in relation.fetchall()]
            )

        return results

    def _write(self, records_by_path: dict[Path, list[dict[str, Any]]]):
        for path, records in records_by_path.items():
            if not records:
                continue

            path = self.__root.joinpath(path)
            typ = PersistableId.parse(records[0]["object_id"]).typ

            self.__fs.makedirs(path.parent, exist_ok=True)

            match self.record_format:
                case RecordFormat.Parquet:
                    schema = self.__schema(self.record_type(typ))
                    table = pa.Table.from_pylist(records, schema=schema)
                    with pq.ParquetWriter(
                        f"{path}.parquet", schema, filesystem=self.__fs
                    ) as writer:
                        writer.write_table(table)
                case RecordFormat.Json:
                    self.__fs.write_bytes(f"{path}.json", dumps(records))
                case _:
                    raise RuntimeError(f"Unsupported format {self.record_format}")
