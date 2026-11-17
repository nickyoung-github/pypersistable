import datetime as dt

import pytest
from models import Cat, Dog
from pydantic import TypeAdapter, ValidationError

from pypersistable import DiscriminatedUnion


def test_single_type_raises():
    with pytest.raises(TypeError, match="multiple types"):
        _ = DiscriminatedUnion[Cat]


def test_value_of_no_member_raises():
    with pytest.raises(ValidationError):
        TypeAdapter(DiscriminatedUnion[Dog, Cat]).validate_python("Cerberus")


def test_untagged_dict_raises():
    with pytest.raises(ValidationError, match="Unable to extract tag"):
        TypeAdapter(DiscriminatedUnion[Dog, Cat]).validate_python({"lives": 1})


@pytest.mark.parametrize("value", [dt.date(2026, 1, 2), "2026-01-02"], ids=type)
def test_non_object_members_tagged(value):
    adapter = TypeAdapter(DiscriminatedUnion[dt.date, str])
    assert adapter.validate_json(adapter.dump_json(value)) == value


def test_untagged_non_object_member_read():
    # As persisted before non-object members were tagged
    assert TypeAdapter(DiscriminatedUnion[Cat, int]).validate_json("5") == 5


def test_json_schema_of_non_object_member():
    schema = TypeAdapter(DiscriminatedUnion[Cat, int]).json_schema()
    assert schema["oneOf"][1] == {
        "type": "object",
        "properties": {
            "__discriminator_type": {"const": "builtins:int"},
            "builtins:int": {"type": "integer"},
        },
        "required": ["__discriminator_type", "builtins:int"],
    }
