"""Tests of FSStore's own behaviour: see test_stores for those of every store"""

import sys
from dataclasses import dataclass
from ipaddress import IPv4Address
from uuid import uuid4

import duckdb
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from models import (
    Counterparty,
    Deal,
    Evolving,
    EvolvingBefore,
    EvolvingRetyped,
    Node,
    TradingBook,
    registry,
)
from pydantic_core import core_schema

from pypersistable import (
    FSStore,
    Identifiable,
    IdField,
    IncompatibleSchemaError,
    ObjectRecord,
    RecordFormat,
    UnknownObjectError,
    UnsupportedTypeError,
    persistable_arrow_converter,
)
from pypersistable.arrow import dataclass_fields


@pytest.fixture(params=list(RecordFormat), ids=lambda f: f.name)
def store(request, tmp_path) -> FSStore:
    return FSStore(f"file://{tmp_path}", record_format=request.param)


@pytest.fixture
def json_store(tmp_path) -> FSStore:
    return FSStore(f"file://{tmp_path}", record_format=RecordFormat.Json)


def test_json_by_value_round_trip(json_store):
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    json_store.write([deal], by_reference=False)

    assert Deal.read(json_store, deal_id="D") == deal
    with pytest.raises(UnknownObjectError):
        Counterparty.read(json_store, name="Acme")  # Embedded, not persisted separately


def test_failed_write_leaves_no_dangling_reference(json_store, monkeypatch):
    write_bytes = LocalFileSystem.write_bytes

    def failing(self, path, *args, **kwargs):
        if "Counterparty" in str(path) and str(path).endswith(".json"):
            raise OSError("Disk full")
        return write_bytes(self, path, *args, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "write_bytes", failing)
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    with pytest.raises(OSError, match="Disk full"):
        deal.write(json_store)
    # Not written, as written after the counterparty it references
    with pytest.raises(UnknownObjectError):
        Deal.read(json_store, deal_id="D")


def test_parquet_records_written_before_active(tmp_path):
    # As written before ObjectRecord._active was added: in no file, so read as its default
    before = persistable_arrow_converter()
    before.add_struct(
        lambda t: (
            {k: v for k, v in (dataclass_fields(t) or {}).items() if k != "_active"}
            if issubclass(t, ObjectRecord)
            else None
        )
    )
    node = Node(name="a")
    node.write(FSStore(f"file://{tmp_path}", arrow_converter=before))
    assert Node.read(FSStore(f"file://{tmp_path}"), name="a") == node


@pytest.mark.skipif(sys.platform == "win32", reason="Windows doesn't support chmod")
def test_unreadable_file_raises(store, tmp_path):
    Counterparty(name="Acme").write(store)
    for path in (tmp_path / "Counterparty").iterdir():
        path.chmod(0)

    try:
        with pytest.raises(duckdb.IOException):
            Counterparty.read(store, name="Acme")
    finally:
        for path in (tmp_path / "Counterparty").iterdir():
            path.chmod(0o644)


def test_dangling_reference_raises(tmp_path):
    store = FSStore(f"file://{tmp_path}", record_format=RecordFormat.Json)
    deal = Deal(
        deal_id="D0", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    store.write([deal])
    for path in (tmp_path / "Counterparty").iterdir():
        path.unlink()

    with pytest.raises(RuntimeError, match="Failed to load references"):
        Deal.read(store, deal_id="D0")


def test_parquet_by_value_raises(tmp_path):
    store = FSStore(f"file://{tmp_path}", record_format=RecordFormat.Parquet)
    deal = Deal(
        deal_id="D", counterparty=Counterparty(name="Acme"), book=TradingBook(name="B")
    )
    with pytest.raises(ValueError, match="by_reference=False is not supported"):
        store.write([deal], by_reference=False)


@pytest.mark.parametrize("record_format", list(RecordFormat), ids=lambda f: f.name)
def test_non_local_filesystem(record_format):
    store = FSStore(f"memory://{uuid4().hex}", record_format=record_format)
    deals = [
        Deal(
            deal_id=f"D{i}",
            counterparty=Counterparty(name="Acme"),
            book=TradingBook(name="B"),
        )
        for i in range(2)
    ]
    store.write(deals)
    assert store.read([d.object_id for d in deals]) == {d.object_id: d for d in deals}


def test_unusual_paths(tmp_path):
    store = FSStore(f"file://{tmp_path}/it's here")
    Node(name="a").write(store)
    assert Node.read(store, name="a") == Node(name="a")


@pytest.mark.parametrize(
    ("hint", "suggestion"),
    [
        (int | str, r"\. Use DiscriminatedUnion"),
        (Counterparty | TradingBook | None, "Use Reference.*or DiscriminatedUnion"),
        (Counterparty | int, r"\. Use DiscriminatedUnion"),
    ],
    ids=str,
)
def test_union_suggestion(hint, suggestion):
    with pytest.raises(UnsupportedTypeError, match=suggestion):
        persistable_arrow_converter().arrow_type(hint)


@dataclass
class Host(Identifiable):
    name: IdField[str]
    address: IPv4Address


registry.register(Host.__name__, Host)


def schema_files(tmp_path) -> list:
    return sorted(tmp_path.rglob("*.jsonschema"))


def test_unrecorded_schema_is_checked(store, redefine, tmp_path):
    # E.g. objects written before schemas were recorded
    Evolving(name="old", added=1).write(store)
    schema_files(tmp_path)[0].unlink()

    redefine(Evolving, EvolvingRetyped)
    with pytest.raises(IncompatibleSchemaError):
        EvolvingRetyped(name="new").write(store)


class Opaque:
    """Validated by a plain function: it has no JSON schema"""

    def __init__(self, value: str):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Opaque) and other.value == self.value

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, Opaque) else Opaque(v),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.value
            ),
        )


@dataclass
class HoldsOpaque(Identifiable):
    name: IdField[str]
    opaque: Opaque


registry.register("HoldsOpaque", HoldsOpaque)


def test_type_without_json_schema(json_store):
    holder = HoldsOpaque(name="h", opaque=Opaque("a"))
    holder.write(json_store)
    assert HoldsOpaque.read(json_store, name="h") == holder


def test_schema_recorded_on_change(store, redefine, tmp_path):
    EvolvingBefore(name="a").write(store)
    EvolvingBefore(name="b").write(store)
    FSStore(f"file://{tmp_path}", record_format=store.record_format).write(
        [EvolvingBefore(name="c")]
    )
    assert len(schema_files(tmp_path)) == 1

    redefine(EvolvingBefore, Evolving)
    Evolving(name="d").write(store)
    assert len(schema_files(tmp_path)) == 2


def test_parquet_custom_type(tmp_path):
    url = f"file://{tmp_path}"
    with pytest.raises(UnsupportedTypeError, match="IPv4Address"):
        Host(name="h", address=IPv4Address("10.0.0.1")).write(FSStore(url))

    converter = persistable_arrow_converter()
    converter.add_type(IPv4Address, pa.string(), str)
    store = FSStore(url, arrow_converter=converter)
    host = Host(name="h", address=IPv4Address("10.0.0.1"))
    host.write(store)
    assert Host.read(store, name="h") == host
