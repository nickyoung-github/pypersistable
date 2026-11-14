from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any, cast

from pydantic import TypeAdapter


def typed_cache[**P, R](
    func: Callable[P, R], maxsize: int | None = None
) -> Callable[P, R]:
    # Useful if you want to cache a derived or abstract property. Without this, pyright complains that the
    # return type of the cached function is not compatible with the property type.
    return cast(Callable[P, R], lru_cache(maxsize=maxsize)(func))


@cache
def type_adapter(typ: Any) -> TypeAdapter:
    """A shared TypeAdapter for typ, as constructing one is expensive"""
    return TypeAdapter(typ)
