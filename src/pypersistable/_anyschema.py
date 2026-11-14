from types import NoneType

from anyschema.exceptions import UnsupportedDTypeError
from anyschema.parsers import (
    AnnotatedStep,
    ForwardRefStep,
    PyTypeStep,
    UnionTypeStep,
    make_pipeline,
)
from anyschema.parsers.annotated_types import AnnotatedTypesStep
from anyschema.parsers.pydantic import PydanticTypeStep
from anyschema.typing import FieldType

from .persistable import Persistable
from .persistable_id import PersistableId


class _UnionTypeStep(UnionTypeStep):
    # References show up as a tuple of the underlying type and PersistableId. This is a hack to handle that.
    # Replacees the anyschema default UnionTypeStep.

    def _parse_union(self, union: tuple[FieldType, ...]) -> FieldType:
        if len(union) != 2:
            if all(isinstance(t, type) and issubclass(t, Persistable) for t in union):
                names = [t.__name__ for t in union if isinstance(t, type)]
                raise UnsupportedDTypeError(
                    f"Union with more than two Persistable types is not supported. Try using Reference[Union[{', '.join(names)}]] instead."
                )
            raise UnsupportedDTypeError("Union with more than two types is not supported.")

        field0, field1 = union

        if field0 is not NoneType and field1 is not NoneType:
            for f in (field0, field1):
                if (
                    f is not PersistableId
                    and isinstance(f, type)
                    and not issubclass(f, Persistable)
                ):
                    raise UnsupportedDTypeError("Union with mixed types is not supported.")

            return str

        return field1 if field0 is NoneType else field0


# Construct our own pipeline, including our custom steps and excluding the default ones we don't care about,
# e.g., for attrs

PARSER_STEPS = (
    ForwardRefStep(),
    _UnionTypeStep(),
    AnnotatedStep(),
    AnnotatedTypesStep(),
    PydanticTypeStep(),
    PyTypeStep(),
)

PIPELINE = make_pipeline(PARSER_STEPS)
