import pytest
from models import Counterparty, TradingBook
from pydantic import TypeAdapter

from pypersistable import PersistableId, UnregisteredTypeError


def test_requires_typ_or_registered_name():
    with pytest.raises(TypeError):
        PersistableId("abc")


def test_str_is_registered_name_and_name():
    assert str(PersistableId("abc", typ=Counterparty)) == "Counterparty/abc"


def test_parse_round_trips_str():
    oid = PersistableId.parse("Counterparty/abc")
    assert (oid.name, oid.registered_name) == ("abc", "Counterparty")
    assert str(oid) == "Counterparty/abc"


def test_resolves_missing_member_lazily():
    assert PersistableId("abc", typ=Counterparty).registered_name == "Counterparty"
    assert PersistableId("abc", registered_name="Counterparty").typ is Counterparty


def test_unregistered_type_fails_on_resolution_only():
    class Unregistered: ...

    oid = PersistableId("abc", typ=Unregistered)
    with pytest.raises(UnregisteredTypeError):
        _ = oid.registered_name
    with pytest.raises(UnregisteredTypeError):
        _ = PersistableId.parse("Unregistered/abc").typ


def test_equal_whether_built_from_typ_or_registered_name():
    by_type = PersistableId("abc", typ=Counterparty)
    parsed = PersistableId.parse("Counterparty/abc")
    assert by_type == parsed
    assert hash(by_type) == hash(parsed)
    assert {by_type: 1}[parsed] == 1


def test_same_name_different_type_not_equal():
    # Names hash only the id field values, so different types can share one
    counterparty = Counterparty(name="Acme").object_id
    book = TradingBook(name="Acme").object_id
    assert counterparty.name == book.name
    assert counterparty != book


def test_not_equal_to_other_types():
    assert PersistableId("abc", typ=Counterparty) != "Counterparty/abc"


def test_pydantic_validates_from_str_and_serialises_to_str():
    adapter = TypeAdapter(PersistableId)
    oid = PersistableId("abc", typ=Counterparty)
    assert adapter.validate_python("Counterparty/abc") == oid
    assert adapter.validate_python(oid) is oid
    assert adapter.dump_python(oid) == "Counterparty/abc"
    assert adapter.dump_json(oid) == b'"Counterparty/abc"'
