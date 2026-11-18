"""A store in a SQL database, using SQLAlchemy and Alembic: install pypersistable[sql]"""

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from hashlib import sha256
from typing import Any

import pyarrow as pa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from orjson import Fragment, dumps
from pydantic import ValidationError
from pydantic_core import to_jsonable_python
from sqlalchemy import (
    JSON,
    REAL,
    BigInteger,
    Boolean,
    Column,
    Connection,
    Date,
    DateTime,
    Double,
    Engine,
    Integer,
    Interval,
    LargeBinary,
    MetaData,
    Numeric,
    RowMapping,
    Select,
    SmallInteger,
    String,
    Table,
    Text,
    Time,
    func,
    insert,
    inspect,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import ColumnElement, cast
from sqlalchemy.types import TypeDecorator, TypeEngine

from .arrow import ArrowConverter, UnsupportedTypeError
from .fs_store import IncompatibleSchemaError, persistable_arrow_converter
from .persistable import Persistable
from .persistable_id import PersistableId
from .persistable_store import WriteConflictError
from .persistable_store_base import (
    ObjectRecord,
    PersistableStoreBase,
    _check_json,
    _DataRecord,
)
from .reference import RESOLVER, ReferenceResolver
from .type_registry import TypeRegistry
from .typed_cache import type_adapter

_RECORD_FIELDS = frozenset(f.name for f in fields(ObjectRecord))
# The data fields of a row which are None, so that its other NULLs are fields absent when it was written, e.g. added
# since: they read as their defaults
_NULLS = "_nulls"
_JSON = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
# An object id, as a key: of bounded length, as some databases require
_KEY = String(255)


class _UTCDateTime(TypeDecorator):
    """A time zone aware datetime, stored as UTC: databases, e.g. SQLite, may not store time zones"""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        return value if value is None else value.astimezone(dt.UTC)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value


class _DecimalText(TypeDecorator):
    """A Decimal, as text: SQLite stores numerics as floats, losing precision"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        return value if value is None else str(value)


def _sql_type(arrow_type: pa.DataType | None) -> TypeEngine:
    """None: a type with no arrow type, as JSON"""
    t = arrow_type
    if t is None:
        return _JSON
    if pa.types.is_boolean(t):
        return Boolean()
    if pa.types.is_integer(t):
        return {8: SmallInteger(), 16: SmallInteger(), 32: Integer()}.get(
            t.bit_width, BigInteger()
        )
    if pa.types.is_floating(t):
        return Double() if t.bit_width == 64 else REAL()
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return Text()
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return LargeBinary()
    if pa.types.is_decimal(t):
        return Numeric(t.precision, t.scale).with_variant(_DecimalText(), "sqlite")
    if pa.types.is_date(t):
        return Date()
    if pa.types.is_time(t):
        return Time()
    if pa.types.is_timestamp(t):
        return _UTCDateTime() if t.tz is not None else DateTime()
    if pa.types.is_duration(t):
        return Interval()
    return _JSON  # Structs, lists and maps: nested objects, collections and DiscriminatedUnions


def _is_json(column: ColumnElement) -> bool:
    return isinstance(column.type, JSON)


class SQLStore(PersistableStoreBase):
    """
    A table per type, named by its registered name, of its records: a column per ObjectRecord field and per field of
    the type, typed by its arrow type, or JSON if nested, and _nulls, the fields which are None. A _latest_versions
    table holds each object's latest version, so that concurrent writes conflict
    """

    def __init__(
        self,
        engine: Engine,
        schema: str | None = None,
        arrow_converter: ArrowConverter | None = None,
        require_load_of_existing: bool = False,
    ):
        """
        engine: e.g. from sqlalchemy.create_engine(url). Its owner disposes of it
        schema: the database schema of the tables, if not the default
        arrow_converter: determines column types: defaults to persistable_arrow_converter()
        """
        super().__init__(require_load_of_existing)
        self.__engine = engine
        self.__schema = schema
        self.__arrow = arrow_converter or persistable_arrow_converter()
        self.__latest = Table(
            "_latest_versions",
            MetaData(schema=schema),
            Column("_object_id", _KEY, primary_key=True),
            Column("_entry_time", _UTCDateTime(), nullable=False),
        )
        self.__tables: dict[type[Persistable], Table] = {}
        # Of the tables known to exist. A redefined type is checked again, as for FSStore
        self.__columns: dict[type[Persistable], Mapping[str, TypeEngine]] = {}
        self.__checked: set[type[Persistable]] = set()

    def __repr__(self) -> str:
        # Its password, if any, hidden
        return f"SQLStore({self.__engine.url.render_as_string()!r})"

    @property
    def _references_by_value(self) -> bool:
        return False

    def __table(self, typ: type[Persistable]) -> Table:
        if (table := self.__tables.get(typ)) is None:
            record = {f.name: f.type for f in self.__arrow.arrow_schema(ObjectRecord)}
            if (hints := self.__arrow.struct_fields(typ)) is None:
                raise UnsupportedTypeError(f"Not a struct-like type: {typ}")
            data = {name: self.__arrow_type(hint) for name, hint in hints.items()}
            if clashes := (_RECORD_FIELDS | {_NULLS}).intersection(data):
                raise ValueError(
                    f"{typ.__qualname__}: fields {', '.join(sorted(clashes))} clash with SQLStore's columns"
                )

            # Shortened, deterministically, if too long for the database, which may otherwise truncate it
            name = TypeRegistry[Persistable]().registered_name(typ)
            limit = self.__engine.dialect.max_identifier_length
            if len(encoded := name.encode()) > limit:
                hashed = sha256(encoded).hexdigest()[:8]
                name = f"{encoded[: limit - 9].decode(errors='ignore')}_{hashed}"

            columns = (
                Column(
                    name,
                    _KEY if name == "_object_id" else _sql_type(arrow_type),
                    primary_key=name in ("_object_id", "_entry_time"),
                    info={"arrow_type": arrow_type},
                )
                for name, arrow_type in {**record, **data}.items()
            )
            table = self.__tables[typ] = Table(
                name, MetaData(schema=self.__schema), *columns, Column(_NULLS, _JSON)
            )
        return table

    def __arrow_type(self, hint: Any) -> pa.DataType | None:
        """hint's arrow type, else None, e.g. for a union of other than T | None: stored as JSON"""
        try:
            return self.__arrow.arrow_type(hint)
        except UnsupportedTypeError:
            return None

    def __persisted(
        self, conn: Connection, typ: type[Persistable]
    ) -> Mapping[str, TypeEngine] | None:
        """The types of the columns of typ's table, None if it does not exist"""
        if (columns := self.__columns.get(typ)) is None:
            name, inspector = self.__table(typ).name, inspect(conn)
            if not inspector.has_table(name, schema=self.__schema):
                return None
            columns = self.__columns[typ] = {
                c["name"]: c["type"]
                for c in inspector.get_columns(name, schema=self.__schema)
            }
        return columns

    def __records(
        self,
        conn: Connection,
        typ: type[Persistable],
        query: Callable[[Table, Sequence[ColumnElement]], Select],
    ) -> list[bytes]:
        """The records of typ which query, given the table and the columns to select, selects, as JSON"""
        if (persisted := self.__persisted(conn, typ)) is None:
            return []

        # Those persisted, e.g. not added since: absent, they read as their defaults. JSON as text, embedded as is
        table = self.__table(typ)
        columns = [
            cast(c, Text).label(c.name) if _is_json(c) and c.name != _NULLS else c
            for c in table.columns
            if c.name in persisted
        ]
        return [
            self.__json(table, row._mapping)
            for row in conn.execute(query(table, columns))
        ]

    @staticmethod
    def __json(table: Table, row: RowMapping) -> bytes:
        nulls = set(row.get(_NULLS) or ())
        record: dict[str, Any] = {}
        data: dict[str, Any] = {}
        for name, value in row.items():
            if name == _NULLS:
                continue
            # A str, not SQLAlchemy's subclass of it, which orjson rejects as a key
            name = str(name)
            target = record if name in _RECORD_FIELDS else data
            if value is not None:
                target[name] = (
                    Fragment(value)
                    if _is_json(table.c[name])
                    else to_jsonable_python(value, bytes_mode="base64")
                )
            elif name in nulls:
                target[name] = None
        return dumps({**record, "_data": data})

    def __row(self, table: Table, record: dict[str, Any]) -> dict[str, Any]:
        data = record["_data"]
        row = dict.fromkeys(table.columns.keys())
        row.update((k, v) for k, v in record.items() if k != "_data")
        row[_NULLS] = [k for k, v in data.items() if v is None] or None
        for column in table.columns:
            if (
                column.name not in _RECORD_FIELDS
                and (value := data.get(column.name)) is not None
            ):
                row[column.name] = (
                    to_jsonable_python(value, bytes_mode="base64")
                    if _is_json(column)
                    else self.__arrow.arrow_value(value, column.info["arrow_type"])
                )
        return row

    def _serialise(
        self, record: _DataRecord, context: dict[str, Any]
    ) -> dict[str, Any]:
        # Read as JSON (see __json), so NaN and infinities would be lost, as would NaN by SQLite, which stores it as NULL
        serialised = super()._serialise(record, context)
        _check_json(serialised)
        return serialised

    def _validate(
        self, typ: type[Persistable], result: Any, resolver: ReferenceResolver
    ) -> _DataRecord:
        return type_adapter(self.record_type(typ)).validate_json(
            result, context={RESOLVER: resolver}, by_name=True
        )

    def _read(
        self,
        by_type: dict[type[Persistable], set[PersistableId]],
        effective_time: dt.datetime,
        entry_time: dt.datetime,
    ) -> dict[type[Persistable], list[bytes]]:
        def latest(ids: set[PersistableId]):
            def query(table: Table, columns: Sequence[ColumnElement]) -> Select:
                where: list[ColumnElement[bool]] = [
                    table.c._object_id.in_([str(i) for i in ids])
                ]
                # Unbounded if max, which, naive, is not comparable with the times persisted
                if effective_time != dt.datetime.max:
                    where.append(table.c._effective_time <= effective_time)
                if entry_time != dt.datetime.max:
                    where.append(table.c._entry_time <= entry_time)
                rank = func.row_number().over(
                    partition_by=table.c._object_id,
                    order_by=table.c._entry_time.desc(),
                )
                ranked = select(*columns, rank.label("_rank")).where(*where).subquery()
                return select(*(ranked.c[c.name] for c in columns)).where(
                    ranked.c._rank == 1
                )

            return query

        # A connection per read, as reads may run concurrently (see ReadBatch)
        with self.__engine.connect() as conn:
            return {
                typ: self.__records(conn, typ, latest(ids))
                for typ, ids in by_type.items()
            }

    def _check_types(self, types: set[type[Persistable]]):
        for typ in types - self.__checked:
            with self.__engine.begin() as conn:
                self.__latest.create(conn, checkfirst=True)
                self.__check_schema(conn, typ)
            self.__checked.add(typ)

    def __check_schema(self, conn: Connection, typ: type[Persistable]):
        """
        On this store's first write of typ: creates its table, else checks it can read the objects persisted, adding
        the columns of fields added since they were written
        """
        table = self.__table(typ)
        self.__columns.pop(typ, None)  # As now, e.g. if since changed by another store
        if (persisted := self.__persisted(conn, typ)) is None:
            table.create(conn)
            return

        dialect = conn.dialect
        changed = [
            f"{c.name} is {ours}, but {theirs} when persisted"
            for c in table.columns
            if c.name in persisted
            and (ours := c.type.compile(dialect))
            != (theirs := persisted[c.name].compile(dialect))
        ]
        if changed:
            raise IncompatibleSchemaError(f"{typ.__qualname__}: {'; '.join(changed)}")

        def latest(table: Table, columns: Sequence[ColumnElement]) -> Select:
            return select(*columns).order_by(table.c._entry_time.desc()).limit(1)

        for result in self.__records(conn, typ, latest):
            try:
                self._validate(typ, result, ReferenceResolver(resolve=False))
            except ValidationError as e:
                raise IncompatibleSchemaError(
                    f"{typ.__qualname__} cannot read its latest object: {e}"
                ) from None

        operations = Operations(MigrationContext.configure(conn))
        for c in table.columns:
            if c.name not in persisted:
                operations.add_column(
                    table.name, Column(c.name, c.type), schema=self.__schema
                )
        self.__columns.pop(typ)

    def _write(
        self,
        batches: Sequence[tuple[type[Persistable], list[dict[str, Any]]]],
        entry_time: dt.datetime,
        expected: Mapping[PersistableId, dt.datetime | None],
    ):
        try:
            # Atomic: all the records are written, or, e.g. if any conflict, none. So the batches' order is immaterial
            with self.__engine.begin() as conn:
                ids = [r["_object_id"] for _, rs in batches for r in rs]
                self.__set_latest(
                    conn, ids, entry_time, {str(i): v for i, v in expected.items()}
                )
                for typ, records in batches:
                    table = self.__table(typ)
                    conn.execute(insert(table), [self.__row(table, r) for r in records])
        except IntegrityError as e:
            # A key already present, e.g. an object first written concurrently
            raise WriteConflictError(f"Written concurrently: {e.orig}") from None

    def __set_latest(
        self,
        conn: Connection,
        ids: list[str],
        entry_time: dt.datetime,
        expected: Mapping[str, dt.datetime | None],
    ):
        """
        Sets the objects' latest versions, if those expected: a compare-and-set, atomic in any database. Also if not
        expected (i.e. forcing an overwrite), only if earlier than entry_time: a version written since this write
        began would otherwise remain the latest
        """
        latest = self.__latest
        key, version = latest.c._object_id, latest.c._entry_time

        replaced = {i: v for i, v in expected.items() if v is not None}
        unchecked = [i for i in ids if i not in expected]
        existing = (
            set(conn.scalars(select(key).where(key.in_(unchecked))))
            if unchecked
            else set()
        )
        if updating := [*replaced, *existing]:
            matching = [key.in_(existing)]
            if replaced:
                matching.append(tuple_(key, version).in_(replaced.items()))
            updated = conn.execute(
                update(latest)
                .where(or_(*matching), version < entry_time)
                .values(_entry_time=entry_time)
            ).rowcount
            if updated != len(updating):
                current = conn.execute(
                    select(key, version).where(key.in_(updating), version != entry_time)
                )
                raise WriteConflictError(
                    "; ".join(
                        f"{i} was written at {t}, during this write" for i, t in current
                    )
                )

        if new := [i for i in ids if expected.get(i) is None and i not in existing]:
            # Raises IntegrityError if any has since been written
            conn.execute(
                insert(latest),
                [{"_object_id": i, "_entry_time": entry_time} for i in new],
            )
