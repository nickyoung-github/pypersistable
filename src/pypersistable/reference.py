from typing import Annotated, Any, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema
from pydantic_core.core_schema import (
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    with_info_wrap_validator_function,
    wrap_serializer_function_ser_schema,
)

from .persistable import Persistable
from .persistable_id import PersistableId

# Context keys
RESOLVER = "resolver"
COLLECTOR = "collector"


class ReferenceResolver:
    """Resolves references when reading.

    An unresolved reference validates as its id and is added to pending, for the caller to load and re-validate.
    If not resolve, references are left as their ids.
    """

    def __init__(self, resolve: bool = True):
        self.resolve = resolve
        self.resolved: dict[PersistableId, Persistable] = {}
        self.pending: set[PersistableId] = set()
        self.misses = 0

    def get(self, object_id: PersistableId) -> Persistable | PersistableId:
        if not self.resolve:
            return object_id
        if (obj := self.resolved.get(object_id)) is not None:
            return obj

        self.pending.add(object_id)
        self.misses += 1
        return object_id


class ReferenceCollector:
    """Collects referenced objects when writing, so that they are persisted too.

    Each object's id is computed once, however often it is referenced. Set referrer to the id of the object being
    serialised to record what it references in graph. Ids of unequal objects sharing them are added to conflicts.
    """

    def __init__(self):
        # By id(obj), which is safe as the objects outlive the write
        self.__ids: dict[int, PersistableId] = {}
        self.__objects: dict[PersistableId, Persistable] = {}
        self.__new: dict[PersistableId, Persistable] = {}
        self.referrer: PersistableId | None = None
        self.graph: dict[PersistableId, set[PersistableId]] = {}
        self.conflicts: set[PersistableId] = set()

    def add(self, obj: Persistable) -> PersistableId:
        if (object_id := self.__ids.get(id(obj))) is None:
            object_id = self.__ids[id(obj)] = obj.object_id
            if (existing := self.__objects.get(object_id)) is None:
                self.__objects[object_id] = self.__new[object_id] = obj
            elif existing != obj:
                self.conflicts.add(
                    object_id
                )  # Not raised here, as pydantic would wrap the error

        if self.referrer is not None:
            self.graph.setdefault(self.referrer, set()).add(object_id)

        return object_id

    def take(self) -> dict[PersistableId, Persistable]:
        """Objects added since the last take, by id"""
        new, self.__new = self.__new, {}
        return new


class _Reference:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type[Persistable], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        schema = handler(source_type)

        def validate(
            value: Any, validator: ValidatorFunctionWrapHandler, info: ValidationInfo
        ) -> Any:
            if not isinstance(value, str | PersistableId):
                return validator(value)  # An instance, or one persisted by value

            resolver: ReferenceResolver | None = (info.context or {}).get(RESOLVER)
            if resolver is None:
                raise RuntimeError(
                    f"Need a context containing {{{RESOLVER!r}: ReferenceResolver}} to resolve references"
                )

            # A persisted reference: not validated as source_type, as it may resolve to its id for now
            return resolver.get(
                value
                if isinstance(value, PersistableId)
                else PersistableId.parse(value)
            )

        def serialise(
            value: Persistable,
            serialiser: SerializerFunctionWrapHandler,
            info: SerializationInfo,
        ) -> Any:
            if isinstance(value, PersistableId):
                return str(value)  # Left unresolved by a ReferenceResolver

            collector: ReferenceCollector | None = (info.context or {}).get(COLLECTOR)
            if collector is None:
                # Serialise by value, via the schema so that any context still applies
                return serialiser(value)

            return str(collector.add(value))

        return with_info_wrap_validator_function(
            validate,
            schema,
            serialization=wrap_serializer_function_ser_schema(
                serialise, schema=schema, info_arg=True
            ),
        )


__REFERENCE = _Reference()
T = TypeVar("T", bound=Persistable)
Reference = Annotated[T, __REFERENCE]
