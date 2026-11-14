import datetime as dt
from enum import Enum, auto
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
from .persistable_store import ObjectRecord, PersistableStore


class FSStoreFormat(Enum):
    Json = auto()
    Parquet = auto()


class FSStore(PersistableStore):
    def __init__(self, url: str, record_format: FSStoreFormat = FSStoreFormat.Parquet):
        self.__fs, root = url_to_fs(url)
        self.__root = Path(root)
        self.__record_format = record_format
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
    ) -> dict[Path, list[dict[str, Any]]]:
        results = {}

        for directory, object_ids in by_directory.items():
            resolved_directory = self.__root.joinpath(directory)
            from_spec = (
                f"""
            FROM read_parquet('{resolved_directory!s}/**/*.parquet', union_by_name = true)
            """
                if self.__record_format is FSStoreFormat.Parquet
                else f"""
            FROM read_json('{resolved_directory!s}/**/*.json', union_by_name = true)
            """
            )

            try:
                table = self.__conn.sql(
                    f"""
                    SELECT *
                    {from_spec}
                    WHERE object_id IN (SELECT unnest(?))
                    AND metadata.effective_time <= ?
                    AND metadata.entry_time <= ?
                    QUALIFY row_number() OVER (PARTITION BY object_id ORDER BY metadata.entry_time DESC) = 1
                    """,
                    params=[[str(i) for i in object_ids], effective_time, entry_time],
                ).to_arrow_table()

                results[directory] = table.to_pylist()
            except duckdb.IOException:
                # This will happen if no objects of this type have been saved
                pass

        return results

    def _write(self, records_by_path: dict[Path, list[dict[str, Any]]]):
        for path, records in records_by_path.items():
            if not records:
                continue

            path = self.__root.joinpath(path)
            typ = PersistableId.parse(records[0]["object_id"]).typ

            self.__fs.makedirs(path.parent, exist_ok=True)

            match self.__record_format:
                case FSStoreFormat.Parquet:
                    schema = self.__schema(self.record_type(typ))
                    table = pa.Table.from_pylist(records, schema=schema)
                    with pq.ParquetWriter(
                        str(path.with_suffix(".parquet")), schema, filesystem=self.__fs
                    ) as writer:
                        writer.write_table(table)
                case FSStoreFormat.Json:
                    self.__fs.write_bytes(path.with_suffix(".json"), dumps(records))
                case _:
                    raise RuntimeError(f"Unsupported format {self.__record_format}")
