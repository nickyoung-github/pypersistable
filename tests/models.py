from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from pypersistable import (
    DiscriminatedUnion,
    Identifiable,
    IdField,
    ObjectStore,
    Reference,
    TypeRegistry,
    Unique,
)

registry = TypeRegistry()


@dataclass(frozen=True, kw_only=True)
class Counterparty(Identifiable):
    name: IdField[str]
    location: str = ""


@dataclass(frozen=True, kw_only=True)
class TradingBook(Identifiable):
    name: IdField[str]


class ModelCounterparty(Identifiable, BaseModel):
    name: IdField[str]
    location: str = ""


@dataclass(frozen=True, kw_only=True)
class Option(Unique):
    ticker: str
    strike: float


class ModelAddress(Unique, BaseModel):
    street: str
    town: str


@dataclass(frozen=True, kw_only=True)
class Thing(Identifiable):
    id: IdField[str]


@dataclass(frozen=True, kw_only=True)
class CustomerBook(Identifiable):
    counterparty: IdField[Counterparty]
    desk: IdField[str]
    coverage: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ReferenceBook(Identifiable):
    counterparty: IdField[Reference[Counterparty]]


class ModelBook(Identifiable, BaseModel):
    counterparty: IdField[ModelCounterparty]


@dataclass(frozen=True, kw_only=True)
class Pair(Unique):
    counterparty: Counterparty
    quantity: int


@dataclass(kw_only=True)
class Scaled(Identifiable):
    name: IdField[str]
    value: int
    scale: InitVar[int] = 1

    def __post_init__(self, scale: int):
        self.value *= scale


class Dog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bark: str


@dataclass(frozen=True)
class Cat:
    lives: int


@dataclass(frozen=True, kw_only=True)
class Kennel(Identifiable):
    name: IdField[str]
    pet: DiscriminatedUnion[Dog, Cat]  # pyright: ignore[reportInvalidTypeForm]
    fee: Decimal = Decimal(0)
    tags: tuple[str, ...] = ()


for _typ in (
    Counterparty,
    TradingBook,
    ModelCounterparty,
    Option,
    ModelAddress,
    Thing,
    CustomerBook,
    ReferenceBook,
    ModelBook,
    Pair,
    Scaled,
    Kennel,
):
    registry.register(_typ.__name__, _typ)


class DictStore(ObjectStore):
    """In-memory store recording each batch of ids it is asked to read."""

    def __init__(self, *objects):
        self.objects = {o.object_id: o for o in objects}
        self.reads = []

    def read(self, object_ids: Iterable, *_times):
        ids = list(object_ids)
        self.reads.append(ids)
        return {i: self.objects[i] for i in ids if i in self.objects}
