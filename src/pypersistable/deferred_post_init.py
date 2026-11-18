"""Defers the post-init hooks of objects which reading has validated before resolving their references: it validates
them again once those are loaded (see PersistableStoreBase.__read)"""

from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .reference import ReferenceResolver

# The resolver of the read validating objects, whose pending references defer their post-init hooks
_RESOLVING: ContextVar["ReferenceResolver | None"] = ContextVar(
    "_RESOLVING", default=None
)


def _deferring(post_init: Any) -> Any:
    # An object's fields are validated before its post-init hook, so its own references are among those pending
    @wraps(post_init)
    def deferring(self: Any, *args: Any, **kwargs: Any):
        resolver = _RESOLVING.get()
        if resolver is None or not resolver.pending:
            post_init(self, *args, **kwargs)

    deferring.deferring = True  # pyright: ignore[reportAttributeAccessIssue]
    return deferring


def defer_post_init(cls: type, name: str):
    """Defers cls's post-init hook, name (its own or inherited), unless deferred already, e.g. inherited"""
    post_init = getattr(cls, name, None)
    if post_init is not None and not getattr(post_init, "deferring", False):
        setattr(cls, name, _deferring(post_init))
