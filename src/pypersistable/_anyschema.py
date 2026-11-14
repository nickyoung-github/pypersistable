from dataclasses import fields, is_dataclass
from typing import Any, get_type_hints

import narwhals as nw
from anyschema.exceptions import UnsupportedDTypeError
from anyschema.parsers import (
    AnnotatedStep,
    ForwardRefStep,
    ParserStep,
    PyTypeStep,
    UnionTypeStep,
    make_pipeline,
)
from anyschema.parsers.annotated_types import AnnotatedTypesStep
from anyschema.parsers.pydantic import PydanticTypeStep
from anyschema.typing import FieldConstraints, FieldMetadata, FieldType
from narwhals.dtypes import DType

from .discriminated_union import TAG_FIELD, DiscriminatedUnionInfo
from .reference import _Reference


def dataclass_fields(typ: type) -> list[tuple[str, Any]]:
    # Unlike anyschema's dataclass adapter, keeps Annotated metadata, such as the Reference marker
    hints = get_type_hints(typ, include_extras=True)
    return [(f.name, hints[f.name]) for f in fields(typ)]


class _ReferenceStep(ParserStep):
    # A Reference is persisted as its object id. Annotated metadata arrives as constraints, whether
    # extracted by AnnotatedStep or from a pydantic FieldInfo

    def parse(
        self,
        input_type: FieldType,
        constraints: FieldConstraints,
        metadata: FieldMetadata,
    ) -> DType | None:
        return (
            nw.String() if any(isinstance(c, _Reference) for c in constraints) else None
        )


class _DiscriminatedUnionStep(ParserStep):
    # A struct of the tag and a nullable field per member: see discriminated_union.STRUCT_LAYOUT

    def parse(
        self,
        input_type: FieldType,
        constraints: FieldConstraints,
        metadata: FieldMetadata,
    ) -> DType | None:
        info = next(
            (c for c in constraints if isinstance(c, DiscriminatedUnionInfo)), None
        )
        if info is None:
            return None

        return nw.Struct(
            [
                nw.Field(TAG_FIELD, nw.String()),
                *(
                    nw.Field(tag, self.pipeline.parse(typ, (), {}))
                    for tag, typ in info.members
                ),
            ]
        )


class _DataclassStep(ParserStep):
    # Replaces anyschema's handling of dataclasses, which drops Annotated metadata

    def parse(
        self,
        input_type: FieldType,
        constraints: FieldConstraints,
        metadata: FieldMetadata,
    ) -> DType | None:
        if not (isinstance(input_type, type) and is_dataclass(input_type)):
            return None

        return nw.Struct(
            [
                nw.Field(name, self.pipeline.parse(hint, (), {}))
                for name, hint in dataclass_fields(input_type)
            ]
        )


class _UnionTypeStep(UnionTypeStep):
    # Only T | None is supported. Explain the alternatives for anything else

    def _parse_union(self, union: tuple[FieldType, ...]) -> FieldType:
        try:
            return super()._parse_union(union)
        except UnsupportedDTypeError:
            raise UnsupportedDTypeError(
                f"Unsupported union {' | '.join(map(str, union))}: only T | None is supported. "
                "Use Reference[...] or DiscriminatedUnion[...] instead"
            ) from None


# Our own pipeline: the steps above take precedence over the defaults, and those we don't need (e.g. attrs) are
# excluded

PIPELINE = make_pipeline(
    (
        ForwardRefStep(),
        _ReferenceStep(),
        _DiscriminatedUnionStep(),
        _UnionTypeStep(),
        AnnotatedStep(),
        AnnotatedTypesStep(),
        PydanticTypeStep(),
        _DataclassStep(),
        PyTypeStep(),
    )
)
