import asyncio
from dataclasses import InitVar, dataclass, fields
from hashlib import sha512

import pytest
from models import (
    Counterparty,
    CustomerBook,
    DictStore,
    ModelAddress,
    ModelBook,
    ModelCounterparty,
    Option,
    Pair,
    ReferenceBook,
    Scaled,
    Thing,
    TradingBook,
)
from orjson import OPT_SORT_KEYS, dumps
from pydantic import TypeAdapter, ValidationError

from pypersistable import Identifiable, IdField, PersistableId, Unique
from pypersistable.object_store import UnknownObjectError


def expected_id(typ: type, payload: dict) -> PersistableId:
    # Pins the id format: changing it orphans everything already persisted
    return PersistableId(
        sha512(dumps(payload, option=OPT_SORT_KEYS)).hexdigest(), typ=typ
    )


# id_type


def test_identifiable_id_type_has_only_id_fields():
    id_type = CustomerBook.id_type()
    assert id_type.__name__ == "CustomerBookId"
    assert {f.name for f in fields(id_type)} == {"counterparty", "desk"}


def test_unique_id_type_is_the_class():
    assert Option.id_type() is Option
    assert ModelAddress.id_type() is ModelAddress


# object_id


@pytest.mark.parametrize(
    ("obj", "payload"),
    [
        (Counterparty(name="Acme", location="NW3"), {"name": "Acme"}),
        (ModelCounterparty(name="Acme", location="NW3"), {"name": "Acme"}),
        (Option(ticker="X", strike=1.5), {"ticker": "X", "strike": 1.5}),
        (
            ModelAddress(street="1 High St", town="T"),
            {"street": "1 High St", "town": "T"},
        ),
        (Thing(id="a"), {"id": "a"}),
    ],
)
def test_object_id_hashes_id_fields(obj, payload):
    assert obj.object_id == expected_id(type(obj), payload)


def test_object_id_ignores_non_id_fields():
    assert (
        Counterparty(name="Acme", location="NW3").object_id
        == Counterparty(name="Acme", location="SW1").object_id
    )
    assert Counterparty(name="Acme").object_id != Counterparty(name="Other").object_id


def test_object_id_from_kwargs_matches_instance():
    counterparty = Counterparty(name="Acme")
    book = CustomerBook(counterparty=counterparty, desk="D", coverage=("x",))
    assert (
        CustomerBook._object_id({"counterparty": counterparty, "desk": "D"})
        == book.object_id
    )
    assert (
        Option._object_id({"ticker": "X", "strike": 1.5})
        == Option(ticker="X", strike=1.5).object_id
    )


def test_non_id_init_var_allowed_on_identifiable():
    scaled = Scaled(name="s", value=2, scale=3)
    assert scaled.value == 6
    assert scaled.object_id == expected_id(Scaled, {"name": "s"})


# Nested Persistables contribute only their object_id


acme_nw3 = Counterparty(name="Acme", location="NW3")
acme_sw1 = Counterparty(name="Acme", location="SW1")
model_nw3 = ModelCounterparty(name="Acme", location="NW3")
model_sw1 = ModelCounterparty(name="Acme", location="SW1")


@pytest.mark.parametrize(
    ("obj", "same_id", "payload"),
    [
        pytest.param(
            CustomerBook(counterparty=acme_nw3, desk="D"),
            CustomerBook(counterparty=acme_sw1, desk="D"),
            {"counterparty": str(acme_nw3.object_id), "desk": "D"},
            id="embedded",
        ),
        pytest.param(
            ReferenceBook(counterparty=acme_nw3),
            ReferenceBook(counterparty=acme_sw1),
            {"counterparty": str(acme_nw3.object_id)},
            id="reference",
        ),
        pytest.param(
            ModelBook(counterparty=model_nw3),
            ModelBook(counterparty=model_sw1),
            {"counterparty": str(model_nw3.object_id)},
            id="base_model",
        ),
        pytest.param(
            Pair(counterparty=acme_nw3, quantity=1),
            Pair(counterparty=acme_sw1, quantity=1),
            {"counterparty": str(acme_nw3.object_id), "quantity": 1},
            id="unique",
        ),
    ],
)
def test_nested_persistable_contributes_only_its_object_id(obj, same_id, payload):
    assert obj.object_id == same_id.object_id
    assert obj.object_id == expected_id(type(obj), payload)


def test_serialisation_without_id_context_is_unchanged():
    book = CustomerBook(counterparty=acme_nw3, desk="D")
    assert TypeAdapter(CustomerBook).dump_python(book, mode="json")["counterparty"] == {
        "name": "Acme",
        "location": "NW3",
    }
    assert ModelBook(counterparty=model_nw3).model_dump(mode="json") == {
        "counterparty": {"name": "Acme", "location": "NW3"}
    }


# Class definition errors


def test_identifiable_without_id_fields_raises():
    with pytest.raises(TypeError, match="does not define any id fields"):

        @dataclass
        class NoIds(Identifiable):
            name: str


def test_id_field_init_var_raises():
    with pytest.raises(TypeError, match="IdField name cannot be an InitVar"):

        @dataclass
        class Inner(Identifiable):
            name: IdField[InitVar[str]]

    with pytest.raises(TypeError, match="IdField name cannot be an InitVar"):

        @dataclass
        class Outer(Identifiable):
            name: InitVar[IdField[str]]


def test_unique_init_var_raises():
    with pytest.raises(TypeError, match="a Unique cannot have InitVars: scale"):

        @dataclass
        class Bad(Unique):
            value: int
            scale: InitVar[int]

    @dataclass
    class Good(Unique):
        value: int

    with pytest.raises(TypeError, match="a Unique cannot have InitVars: scale"):

        @dataclass
        class BadSubclass(Good):
            scale: InitVar[int]


# read


def test_read_returns_object():
    counterparty = Counterparty(name="Acme", location="NW3")
    store = DictStore(counterparty)
    assert Counterparty.read(store, name="Acme") == counterparty


def test_read_unknown_object_raises():
    with pytest.raises(UnknownObjectError, match=r"Counterparty\(name='Nobody'\)"):
        Counterparty.read(DictStore(), name="Nobody")


def test_read_rejects_extra_and_missing_id_fields():
    with pytest.raises(ValidationError, match="Unexpected keyword argument"):
        Counterparty.read(DictStore(), name="Acme", location="NW3")
    with pytest.raises(ValidationError, match="Field required"):
        Counterparty.read(DictStore())


def test_async_reads_are_batched_into_one_store_read():
    counterparty, book = Counterparty(name="Acme"), TradingBook(name="Acme")
    store = DictStore(counterparty, book)

    async def read_both():
        return await asyncio.gather(
            Counterparty.read(store, is_async=True, name="Acme"),
            TradingBook.read(store, is_async=True, name="Acme"),
        )

    assert asyncio.run(read_both()) == [counterparty, book]
    assert len(store.reads) == 1
    assert set(store.reads[0]) == {counterparty.object_id, book.object_id}
