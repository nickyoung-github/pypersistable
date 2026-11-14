from typing import Any, Self

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from .type_registry import TypeRegistry


class PersistableId:
    """The id of a persisted object: name (the hash of its id fields) and its type.

    At least one of typ and registered_name must be supplied; the other is resolved
    lazily from the TypeRegistry."""

    __slots__ = ("__registered_name", "__typ", "name")

    def __init__(
        self, name: str, *, typ: type | None = None, registered_name: str | None = None
    ):
        if typ is None and registered_name is None:
            raise TypeError("One of typ or registered_name must be supplied")

        self.name = name
        self.__typ = typ
        self.__registered_name = registered_name

    @classmethod
    def parse(cls, value: str) -> Self:
        registered_name, _, name = value.rpartition("/")
        return cls(name, registered_name=registered_name)

    @property
    def typ(self) -> type:
        if self.__typ is None:
            self.__typ = TypeRegistry().get_type(self.registered_name)
        return self.__typ

    @property
    def registered_name(self) -> str:
        if self.__registered_name is None:
            self.__registered_name = TypeRegistry().registered_name(self.typ)
        return self.__registered_name

    def __eq__(self, other: object) -> bool:
        # Names are hashes of the id field values only, so different types can share a name
        return (
            isinstance(other, PersistableId)
            and self.name == other.name
            and self.registered_name == other.registered_name
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return f"{self.registered_name}/{self.name}"

    def __repr__(self) -> str:
        return f"PersistableId({self!s})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: v if isinstance(v, cls) else cls.parse(v),
            serialization=core_schema.to_string_ser_schema(when_used="always"),
        )
