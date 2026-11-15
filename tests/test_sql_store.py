"""Tests of SQLStore's own behaviour: see test_stores for those of every store"""

from dataclasses import dataclass
from ipaddress import IPv4Address

import pyarrow as pa
import pytest
from conftest import STORE_KINDS
from models import Counterparty, Deal, Node, TradingBook, registry
from sqlalchemy import create_engine
from sqlalchemy.dialects import sqlite

from pypersistable import (
    ArrowConverter,
    Identifiable,
    IdField,
    UnknownObjectError,
    UnsupportedTypeError,
    WriteConflictError,
)
from pypersistable.sql_store import SQLStore, _sql_type


@pytest.fixture(params=[k for k in STORE_KINDS if k in ("SQLite", "PostgreSQL")])
def store_kind(request) -> str:
    return request.param


@dataclass(kw_only=True)
class NoneNotDefault(Identifiable):
    name: IdField[str]
    value: int | None = 5


@dataclass(kw_only=True)
class Clashing(Identifiable):
    name: IdField[str]
    _nulls: int = 0


@dataclass(kw_only=True)
class NoArrowTypes(Identifiable):
    name: IdField[str]
    address: IPv4Address
    values: list[int | str]
    count: int = 0


@dataclass(kw_only=True)
class LongNamedA(Identifiable):
    name: IdField[str]


@dataclass(kw_only=True)
class LongNamedB(Identifiable):
    name: IdField[str]


registry.register("NoneNotDefault", NoneNotDefault)
registry.register("Clashing", Clashing)
registry.register("NoArrowTypes", NoArrowTypes)
# Too long for PostgreSQL, which would truncate both to the same name
registry.register(f"{'Long' * 20}A", LongNamedA)
registry.register(f"{'Long' * 20}B", LongNamedB)


def concurrently(store: SQLStore, monkeypatch, write):
    """Calls write during store's next write, after it has checked for conflicts, before it writes"""
    _write = store._write

    def interleaved(*args):
        write()
        _write(*args)

    monkeypatch.setattr(store, "_write", interleaved)


def test_concurrent_write_conflicts(make_store, monkeypatch):
    store, other = make_store(), make_store()
    Node(name="a").write(store)
    mine, theirs = Node.read(store, name="a"), Node.read(other, name="a")

    theirs.next = Node(name="b")
    concurrently(store, monkeypatch, lambda: theirs.write(other))
    mine.next = Node(name="c")
    with pytest.raises(WriteConflictError, match="Node/.* was written at .* during"):
        mine.write(store)

    # Nothing written
    assert Node.read(store, name="a") == theirs
    with pytest.raises(UnknownObjectError):
        Node.read(store, name="c")


def test_concurrent_first_write_conflicts(make_store, monkeypatch):
    store, other = make_store(), make_store()
    Node(name="x").write(store)  # Creates the table
    concurrently(store, monkeypatch, lambda: Node(name="a").write(other))
    with pytest.raises(WriteConflictError, match="Written concurrently"):
        Node(name="a", next=Node(name="b")).write(store)


def test_concurrent_write_force_overwrite(make_store, monkeypatch):
    store, other = make_store(), make_store()
    Node(name="a").write(store)
    mine = Node.read(store, name="a")

    concurrently(
        store, monkeypatch, lambda: Node(name="a", next=Node(name="b")).write(other)
    )
    mine.next = Node(name="c")
    # Begun before the concurrent write, so of an earlier entry time: it cannot overwrite it
    with pytest.raises(WriteConflictError, match="Node/.* was written at .* during"):
        mine.write(store, force_overwrite=True)
    assert Node.read(store, name="a") != mine

    # Again: now found by its check
    with pytest.warns(UserWarning, match="Overwriting"):
        mine.write(store, force_overwrite=True)
    assert Node.read(store, name="a") == mine


def test_by_value_raises(store):
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    with pytest.raises(ValueError, match=r"not supported by SQLStore\("):
        store.write([deal], by_reference=False)


def test_none_is_not_absent(store):
    # NULL, as for an absent field (which reads as its default), but recorded as None
    obj = NoneNotDefault(name="n", value=None)
    obj.write(store)
    assert NoneNotDefault.read(store, name="n").value is None


def test_fields_without_arrow_types_are_json(store):
    obj = NoArrowTypes(name="n", address=IPv4Address("10.0.0.1"), values=[1, "a"])
    obj.write(store)
    read = NoArrowTypes.read(store, name="n")
    assert read == obj
    assert type(read.address) is IPv4Address


def test_not_struct_like_raises(tmp_path):
    class NoStructs(ArrowConverter):
        # E.g. lacking the add_struct for attrs classes which a type is
        def struct_fields(self, typ):
            return None if issubclass(typ, Node) else super().struct_fields(typ)

    engine = create_engine(f"sqlite:///{tmp_path}/store.db")
    with pytest.raises(UnsupportedTypeError, match="Not a struct-like type"):
        Node(name="a").write(SQLStore(engine, arrow_converter=NoStructs()))
    engine.dispose()


def test_clashing_field_raises(store):
    with pytest.raises(ValueError, match="fields _nulls clash"):
        Clashing(name="c").write(store)


def test_long_table_names(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/store.db")
    engine.dialect.max_identifier_length = 63  # As for PostgreSQL
    store = SQLStore(engine)
    a, b = LongNamedA(name="x"), LongNamedB(name="x")
    store.write([a, b])
    assert store.read([a.object_id, b.object_id]) == {a.object_id: a, b.object_id: b}
    engine.dispose()


def test_repr():
    # Its password, if any, hidden
    engine = create_engine("postgresql+psycopg://user:secret@host/db")
    assert repr(SQLStore(engine)) == "SQLStore('postgresql+psycopg://user:***@host/db')"


@pytest.mark.parametrize(
    ("arrow_type", "sql_type"),
    [
        (pa.int8(), "SMALLINT"),
        (pa.int32(), "INTEGER"),
        (pa.int64(), "BIGINT"),
        (pa.float32(), "REAL"),
        (pa.float64(), "DOUBLE"),
        (pa.large_binary(), "BLOB"),
        (pa.struct([("a", pa.int64())]), "JSON"),
    ],
    ids=str,
)
def test_column_types(arrow_type, sql_type):
    assert _sql_type(arrow_type).compile(sqlite.dialect()) == sql_type
