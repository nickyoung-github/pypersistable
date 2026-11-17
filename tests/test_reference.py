import pytest
from models import Counterparty, TradingBook
from pydantic import TypeAdapter, ValidationError

from pypersistable import Reference
from pypersistable.reference import RESOLVER, ReferenceResolver


def test_resolved_reference_validated():
    book, counterparty = TradingBook(name="B"), Counterparty(name="Acme")
    resolver = ReferenceResolver()
    resolver.resolved = {book.object_id: book, counterparty.object_id: counterparty}
    adapter = TypeAdapter(Reference[Counterparty])

    def validate(obj):
        return adapter.validate_python(str(obj.object_id), context={RESOLVER: resolver})

    assert validate(counterparty) is counterparty
    # A reference to an object of another type
    with pytest.raises(ValidationError):
        validate(book)
