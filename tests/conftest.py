import os
from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from pypersistable import FSStore, PersistableStoreBase, RecordFormat, TypeRegistry
from pypersistable.sql_store import SQLStore

# A PostgreSQL database to test SQLStore with too, e.g. postgresql+psycopg://postgres@localhost/postgres. Each test
# uses its own schema in it
POSTGRES_URL = os.environ.get("PYPERSISTABLE_TEST_POSTGRES_URL")
STORE_KINDS = ("Json", "Parquet", "SQLite", *(("PostgreSQL",) if POSTGRES_URL else ()))


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    assert POSTGRES_URL is not None
    engine = create_engine(POSTGRES_URL)
    yield engine
    engine.dispose()


@pytest.fixture(params=STORE_KINDS)
def store_kind(request) -> str:
    return request.param


@pytest.fixture
def make_store(
    store_kind, tmp_path, request
) -> Iterator[Callable[..., PersistableStoreBase]]:
    """Makes stores of store_kind sharing their storage, given their other arguments"""
    if store_kind in ("Json", "Parquet"):
        yield lambda **kwargs: FSStore(
            f"file://{tmp_path}", record_format=RecordFormat[store_kind], **kwargs
        )
    elif store_kind == "SQLite":
        engine = create_engine(f"sqlite:///{tmp_path}/store.db")
        yield lambda **kwargs: SQLStore(engine, **kwargs)
        engine.dispose()
    else:
        engine = request.getfixturevalue("postgres_engine")
        schema = f"test_{uuid4().hex}"
        with engine.begin() as conn:
            conn.execute(CreateSchema(schema))
        yield lambda **kwargs: SQLStore(engine, schema=schema, **kwargs)
        with engine.begin() as conn:
            conn.execute(DropSchema(schema, cascade=True))


@pytest.fixture
def store(make_store) -> PersistableStoreBase:
    return make_store()


@pytest.fixture
def redefine(monkeypatch):
    """Makes a type a redefinition of another: persisted as it, as if its code had been changed"""

    def redefine(old: type, new: type):
        for method in ("registered_name", "root"):
            original = getattr(TypeRegistry, method)
            monkeypatch.setattr(
                TypeRegistry,
                method,
                lambda self, t, original=original: original(
                    self, old if t is new else t
                ),
            )

    return redefine
