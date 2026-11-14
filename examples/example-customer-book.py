from dataclasses import dataclass  # noqa: N999

from pypersistable import (
    FSStore,
    Identifiable,
    IdField,
    RecordFormat,
    Reference,
    TypeRegistry,
)

registry = TypeRegistry()


@dataclass(kw_only=True)
class Counterparty(Identifiable):
    name: IdField[str]
    location: str
    contact_name: str


registry.register("Counterparty", Counterparty)


@dataclass(kw_only=True)
class CustomerBook(Identifiable):
    counterparty: IdField[Reference[Counterparty]]
    sales_coverage: tuple[str, ...]


registry.register("CustomerBook", CustomerBook)


ctpy = Counterparty(name="Acme", location="NW3", contact_name="Nick")
book = CustomerBook(counterparty=ctpy, sales_coverage=("Tweedledum", "Tweedledee"))
root = r"file:///Users/Nick/ObjectStore"

for store_format in (RecordFormat.Json, RecordFormat.Parquet):
    store = FSStore(f"{root}/{store_format.name.lower()}", record_format=store_format)
    # This will cause both book and ctpy (because it is a reference) to be written
    book.write(store)

    ctpy_loaded, book_loaded = store.read_all(
        Counterparty.read(store, is_async=True, name="Acme"),
        CustomerBook.read(store, is_async=True, counterparty=ctpy),
    )

    print(f"({store_format!s}) Loaded counterparty: {ctpy_loaded}")
    print(
        f"({store_format!s})Loaded counterparty matches original: {ctpy_loaded == ctpy}\n"
    )

    book_id = book.object_id
    book_loaded = store.read([book_id])[book_id]
    print(f"({store_format!s})Loaded customer book: {book_loaded}")
    print(
        f"({store_format!s})Loaded customer book matches original: {book_loaded == book}\n"
    )
