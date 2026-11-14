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
