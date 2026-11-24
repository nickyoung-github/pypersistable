"""Queries of the objects in a store, as SQLAlchemy conditions on their fields: install pypersistable[sql]

    db.exec(select(Trade).where(Trade.counterparty.parent.country == "Belgium", Trade.notional > 1e6))

The fields of a registered Persistable class are, as attributes of the class, its query columns: SQLAlchemy's, so with
its operators. Those of a field holding an object, embedded or by Reference, are the object's fields.
"""

import datetime as dt
from collections.abc import Callable, Mapping
from functools import cache
from types import NoneType, UnionType
from typing import Annotated, Any, Self, Union, cast, get_args, get_origin

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.sql import operators, visitors
from sqlalchemy.types import NullType

from .arrow import ArrowConverter, dataclass_fields, pydantic_fields
from .persistable import Persistable
from .reference import _Reference


@cache
def _hints(typ: type) -> Mapping[str, Any]:
    """The type hints of the fields of a struct-like type: none if it is not one"""
    return dataclass_fields(typ) or pydantic_fields(typ) or {}


def _target(hint: Any) -> tuple[type | None, bool]:
    """The type of the object which a field of hint holds, if one struct-like type, and whether it is a Reference"""
    metadata: tuple[Any, ...] = ()
    while True:
        origin = get_origin(hint)
        if origin is Annotated:
            hint, *extra = get_args(hint)
            metadata += tuple(extra)
        elif (
            origin in (Union, UnionType)
            and len(members := [a for a in get_args(hint) if a is not NoneType]) == 1
        ):
            hint = members[0]
        else:
            break
    return (
        hint if isinstance(hint, type) and _hints(hint) else None,
        any(isinstance(m, _Reference) for m in metadata),
    )


class _FieldType(NullType):
    """The type of a FieldColumn, and so of the values compared with it, until a store builds the query"""

    def __init__(self, column: "FieldColumn"):
        self.column = column

    def coerce_compared_value(self, op: Any, value: Any) -> Self:
        return self


class FieldColumn(sa.ColumnClause):
    """The field at path from root, a Persistable class: e.g. Trade.counterparty.country"""

    inherit_cache = True

    def __init__(self, root: type, path: tuple[str, ...]):
        super().__init__(".".join(path), _FieldType(self))
        self.root = root
        self.path = path


class _ObjectField:
    """A field holding an object: its attributes are the object's fields. One referenced compares, by its id"""

    def __init__(self, root: type, typ: type, path: tuple[str, ...], reference: bool):
        self.__root = root
        self.__typ = typ
        self.__path = path
        self.__reference = reference

    def __getattr__(self, name: str) -> Any:
        return _attribute(self.__root, self.__typ, (*self.__path, name))

    def __column(self, compared: bool = True) -> FieldColumn:
        if compared and not self.__reference:
            raise TypeError(
                f"{'.'.join(self.__path)} is an embedded {self.__typ.__qualname__}: compare its fields"
            )
        return FieldColumn(self.__root, self.__path)

    def __eq__(self, other: object) -> Any:
        return self.__column() == other

    def __ne__(self, other: object) -> Any:
        return self.__column() != other

    def in_(self, others: Any) -> Any:
        return self.__column().in_(others)

    def is_(self, other: None) -> Any:
        return self.__column(compared=False).is_(other)

    def is_not(self, other: None) -> Any:
        return self.__column(compared=False).is_not(other)


def _attribute(root: type, typ: type, path: tuple[str, ...]) -> Any:
    """The query attribute of the field path[-1] of typ, at path from root"""
    try:
        hint = _hints(typ)[path[-1]]
    except KeyError:
        raise AttributeError(f"{typ.__qualname__} has no field {path[-1]!r}") from None
    target, reference = _target(hint)
    if target is None:
        return FieldColumn(root, path)
    return _ObjectField(root, target, path, reference)


class Select[T: Persistable]:
    """The objects of typ, a registered Persistable class, meeting all its conditions: see PersistableStore.exec"""

    def __init__(
        self, typ: type[T], conditions: tuple[sa.ColumnElement[bool], ...] = ()
    ):
        self.typ = typ
        self.conditions = conditions

    def where(self, *conditions: sa.ColumnElement[bool] | bool) -> Self:
        """conditions: on the fields of typ, e.g. Trade.notional > 1e6. To type checkers, which take a class's fields
        to be of their types, they are bools: see col"""
        for condition in conditions:
            if not isinstance(condition, sa.ColumnElement):
                # E.g. a bool, of the default of a field of a class not registered
                raise TypeError(
                    f"Not a condition on the fields of a registered class: {condition!r}"
                )
        return type(self)(self.typ, (*self.conditions, *conditions))  # pyright: ignore[reportArgumentType]


select = Select


def col[T](field: T) -> sa.ColumnElement[T]:
    """A class's field, as type checkers should take it to be, e.g. col(Counterparty.country).in_(["USA", "UK"])"""
    return cast(Any, field)


def _value(arrow: ArrowConverter, hint: Any, value: Any) -> Any:
    """A value compared with a field of hint, as the field is persisted"""
    if value is None:
        return None
    target, reference = _target(hint)
    if reference:
        # Or its id
        return str(value.object_id if isinstance(value, Persistable) else value)
    if target is not None:
        raise TypeError(
            f"Cannot compare {value!r} with an embedded {target.__qualname__}: compare its fields"
        )
    return arrow.arrow_value(value, arrow.arrow_type(hint))


def build[T: Persistable](
    statement: Select[T],
    records: Callable[[type], sa.FromClause | None],
    field: Callable[[sa.FromClause, type, tuple[str, ...], Any], sa.ColumnElement],
    arrow: ArrowConverter,
    effective_time: dt.datetime,
    entry_time: dt.datetime,
) -> sa.Select | None:
    """A query of the ids of the objects statement selects: those whose latest versions, as of the times, meet its
    conditions, as do those of the objects they reference. None if no objects of its type are persisted.

    records: the persisted records of a type, with columns _object_id, _effective_time and _entry_time: None if none
    field: a column of the latest of those records, given them, their type, a path within it (of a field of it, or of
        an object embedded in it) and the field's type hint: null if not persisted
    arrow: converts the values compared with fields, as when written
    """

    def latest(typ: type) -> sa.Subquery | None:
        if (persisted := records(typ)) is None:
            return None
        c = persisted.c
        # Unbounded if max, which, naive, is not comparable with the times persisted
        bounds = [
            column <= time
            for column, time in (
                (c._effective_time, effective_time),
                (c._entry_time, entry_time),
            )
            if time != dt.datetime.max
        ]
        rank = sa.func.row_number().over(
            partition_by=c._object_id, order_by=c._entry_time.desc()
        )
        ranked = sa.select(persisted, rank.label("_rank")).where(*bounds).subquery()
        return sa.select(ranked).where(ranked.c._rank == 1).subquery()

    if (root := latest(statement.typ)) is None:
        return None

    source: sa.FromClause = root
    # The latest records of the objects referenced, by the path of the reference: None if none are persisted
    referenced: dict[tuple[str, ...], sa.Subquery | None] = {}
    resolved: dict[tuple[str, ...], tuple[sa.ColumnElement, Any]] = {}

    def resolve(column: FieldColumn) -> tuple[sa.ColumnElement, Any]:
        """A field's column, and its type hint"""
        nonlocal source
        if not issubclass(statement.typ, column.root):
            raise TypeError(
                f"{column.root.__qualname__}.{column.name} is not a field of {statement.typ.__qualname__}"
            )
        if (path := column.path) not in resolved:
            # The field at local within owner, whose latest records are latest_records
            latest_records, owner, local = root, statement.typ, ()
            typ, hint = owner, None
            for i, name in enumerate(path):
                hint = _hints(typ)[name]
                local = (*local, name)
                target, reference = _target(hint)
                if target is None:
                    break
                typ = target
                if reference and i < len(path) - 1:
                    # A left join per reference followed. A latest version per id, so it adds no rows
                    if (prefix := path[: i + 1]) not in referenced:
                        joined = referenced[prefix] = (
                            None if latest_records is None else latest(typ)
                        )
                        if latest_records is not None and joined is not None:
                            source = source.outerjoin(
                                joined,
                                field(latest_records, owner, local, hint)
                                == joined.c._object_id,
                            )
                    latest_records, owner, local = referenced[prefix], typ, ()
            resolved[path] = (
                sa.null()
                if latest_records is None
                else field(latest_records, owner, local, hint),
                hint,
            )
        return resolved[path]

    def replace(element: Any, **_: Any) -> Any:
        if (
            isinstance(element, sa.BinaryExpression)
            and isinstance(right := element.right, sa.BindParameter)
            and right.expanding
            and not right.value
        ):
            # In no values, which a database may be unable to type
            return sa.true() if element.operator is operators.not_in_op else sa.false()
        if isinstance(element, FieldColumn):
            return resolve(element)[0]
        if isinstance(element, sa.BindParameter) and isinstance(
            element.type, _FieldType
        ):
            # A value compared with a field: converted, and of the field's type
            column, hint = resolve(element.type.column)
            value: Any = element.value
            return sa.bindparam(
                None,
                [_value(arrow, hint, v) for v in value]
                if element.expanding
                else _value(arrow, hint, value),
                type_=column.type,
                expanding=element.expanding,
                unique=True,
            )
        return None

    conditions = [
        visitors.replacement_traverse(c, {}, replace) for c in statement.conditions
    ]
    return sa.select(root.c._object_id).select_from(source).where(*conditions)


_DUCKDB = PGDialect(paramstyle="qmark")


def duckdb_sql(query: sa.Select) -> tuple[str, list[Any]]:
    """query as duckdb's SQL, whose dialect is PostgreSQL's, and its parameters"""
    compiled = query.compile(
        dialect=_DUCKDB, compile_kwargs={"render_postcompile": True}
    )
    return str(compiled), [compiled.params[k] for k in compiled.positiontup or ()]
