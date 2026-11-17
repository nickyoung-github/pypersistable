from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Union

from pydantic import (
    BeforeValidator,
    Discriminator,
    GetJsonSchemaHandler,
    SerializationInfo,
    Tag,
    WrapSerializer,
)
from pydantic_core import CoreSchema


class _TagJsonSchema:
    def __init__(self, tag_field: str, tag_value: str):
        self.__tag_field = tag_field
        self.__tag_value = tag_value

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ):
        json_schema = handler(schema)
        resolved = handler.resolve_ref_schema(json_schema)

        if resolved.get("type") != "object":
            # Handle the case where the schema is not an object, e.g. a literal or enum
            return {
                "type": "object",
                "properties": {
                    self.__tag_field: {"const": self.__tag_value},
                    self.__tag_value: json_schema,
                },
                "required": [self.__tag_field, self.__tag_value],
            }

        return {
            **resolved,
            "properties": {
                **resolved.get("properties", {}),
                self.__tag_field: {"const": self.__tag_value},
            },
            "required": [*resolved.get("required", []), self.__tag_field],
        }


TAG_FIELD = "__discriminator_type"  # Should not clash with any actual field names

# Context key requesting serialisation as a struct, {TAG_FIELD: tag, tag: data}, rather than the payload tagged with
# TAG_FIELD. For formats with no union type, e.g. Parquet: each member becomes a nullable field of the struct
STRUCT_LAYOUT = "discriminated_union_struct_layout"


@dataclass(frozen=True)
class DiscriminatedUnionInfo:
    """Annotated metadata describing a DiscriminatedUnion, for schema generation"""

    members: tuple[tuple[str, type], ...]  # (tag, type)


def tag(typ: type) -> str:
    return f"{typ.__module__}:{typ.__qualname__}"


if TYPE_CHECKING:
    # For type checkers, a union of its members
    DiscriminatedUnion = Union
else:

    class DiscriminatedUnion:
        def __class_getitem__(cls, params: tuple[type, ...]) -> Any:
            if not isinstance(params, tuple | list) or len(params) < 2:
                raise TypeError("Can only use with multiple types")

            def strip_tag(value: Any) -> Any:
                if isinstance(value, dict):
                    if (value_tag := value.get(TAG_FIELD)) in value:
                        return value[value_tag]
                    return {k: v for k, v in value.items() if k != TAG_FIELD}
                return value

            def discriminate(value: Any) -> str | None:
                if isinstance(value, dict):
                    return value.get(TAG_FIELD)
                elif isinstance(value, params):
                    return tag(type(value))
                return None

            def serialise(value: Any, handler: Any, info: SerializationInfo) -> Any:
                data = handler(value)
                value_tag = tag(type(value))
                if (info.context or {}).get(STRUCT_LAYOUT):
                    return {TAG_FIELD: value_tag, value_tag: data}
                return (
                    {TAG_FIELD: value_tag, **data}
                    if isinstance(data, dict)
                    else {TAG_FIELD: value_tag, value_tag: data}
                )

            union_members = tuple(
                Annotated[
                    t,
                    BeforeValidator(strip_tag),
                    Tag(tag(t)),
                    _TagJsonSchema(TAG_FIELD, tag(t)),
                ]
                for t in params
            )

            return Annotated[
                Union[union_members],  # noqa: UP007
                Discriminator(discriminate),
                WrapSerializer(serialise),
                DiscriminatedUnionInfo(tuple((tag(t), t) for t in params)),
            ]
