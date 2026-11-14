"""Persistable classes whose fields are validated by ValidatingDataclassMeta"""

import dataclasses
from dataclasses import dataclass

import pytest
from models import registry
from pydantic import ValidationError

from pypersistable import (
    FSStore,
    Identifiable,
    IdField,
    RecordFormat,
    Reference,
    Unique,
    UnknownObjectError,
    ValidatingDataclassMeta,
)


@dataclass(kw_only=True)
class VLeaf(Identifiable, metaclass=ValidatingDataclassMeta):
    name: IdField[str]
    value: int = 0


@dataclass(kw_only=True)
class VBranch(Identifiable, metaclass=ValidatingDataclassMeta):
    name: IdField[str]
    leaf: Reference[VLeaf]
    leaves: tuple[Reference[VLeaf], ...] = ()
    parent: "Reference[VBranch] | None" = None


@dataclass(slots=True, kw_only=True)
class VSlotsLeaf(Identifiable, metaclass=ValidatingDataclassMeta):
    name: IdField[str]
    value: int = 0


@dataclass(slots=True, kw_only=True)
class VSlotsBranch(Identifiable, metaclass=ValidatingDataclassMeta):
    name: IdField[str]
    leaf: Reference[VSlotsLeaf]
    leaves: tuple[Reference[VSlotsLeaf], ...] = ()
    parent: "Reference[VSlotsBranch] | None" = None


@dataclass(frozen=True, slots=True, kw_only=True)
class VById(Identifiable, metaclass=ValidatingDataclassMeta):
    leaf: IdField[Reference[VSlotsLeaf]]


@dataclass(frozen=True, kw_only=True)
class VUnique(Unique, metaclass=ValidatingDataclassMeta):
    x: int


for _typ in (VLeaf, VBranch, VSlotsLeaf, VSlotsBranch, VById, VUnique):
    registry.register(_typ.__name__, _typ)


@pytest.fixture(params=list(RecordFormat), ids=lambda f: f.name)
def store(request, tmp_path) -> FSStore:
    return FSStore(f"file://{tmp_path}", record_format=request.param)


@pytest.fixture(
    params=((VLeaf, VBranch), (VSlotsLeaf, VSlotsBranch)), ids=("dict", "slots")
)
def models(request) -> tuple[type, type]:
    return request.param


def test_round_trip(store, models):
    leaf_type, branch_type = models
    leaf = leaf_type(name="l", value=1)
    root = branch_type(name="root", leaf=leaf)
    branch = branch_type(name="b", leaf=leaf, leaves=(leaf,), parent=root)
    branch.write(store)

    read = branch_type.read(store, name="b")
    assert read == branch
    assert read.object_entry_time is not None
    assert dataclasses.replace(read, name="b") == branch


def test_assign_and_rewrite(store, models):
    leaf_type, branch_type = models
    branch_type(name="b", leaf=leaf_type(name="l")).write(store)

    read = branch_type.read(store, name="b")
    read.leaf = leaf_type(name="l", value=2)
    read.write(store)
    assert branch_type.read(store, name="b").leaf.value == 2


def test_assignment_validated(models):
    leaf_type, branch_type = models
    branch = branch_type(name="b", leaf=leaf_type(name="l"))
    with pytest.raises(ValidationError):
        branch.leaf = 1
    # A string is a persisted reference's id, which only reading resolves
    with pytest.raises(ValidationError, match="is a persisted reference"):
        branch.leaf = "VLeaf/abc"
    with pytest.raises(ValidationError):
        leaf_type(name="l", value="1")


def test_reference_id_field(store):
    by_id = VById(leaf=VSlotsLeaf(name="l"))
    by_id.write(store)
    assert VById.read(store, leaf=VSlotsLeaf(name="l")) == by_id


def test_unique(store):
    unique = VUnique(x=1)
    unique.write(store)
    assert store.read([unique.object_id]) == {unique.object_id: unique}


def test_delete_by_id_fields(store, models):
    leaf_type, _ = models
    leaf_type(name="l").write(store)
    leaf_type.delete(store, name="l")
    with pytest.raises(UnknownObjectError):
        leaf_type.read(store, name="l")
