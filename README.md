# pypersistable: low effort persistence of python objects

`pypersistable` provides a mechanism for easily persisting python objects, with as little prescription as possible 
as to how they are defined. It is primarily aimed at object models defined using the standard library `dataclass`.

Any type that `pydantic.TypeAdapter` can handle (dataclasses, pydantic models etc.) can be made persistable.
Objects are persisted as JSON or Parquet (with some caveats, see [Parquet](#parquet)). It could likely be extended
to handle `attrs`, [msgspec](https://pypi.org/project/msgspec) etc quite easily.

There are some components, such as [DiscriminatedUnion](#discriminatedunion), [ArrowConverter](#parquet)
and [ValidatingDataclassMeta](#validating-dataclasses), which can be used standalone. These may be moved out to separate packages in future.

## Defining persistable objects

### `TypeRegistry`
Persistable types must be registered with `TypeRegistry`, a singleton:

```python
from pathlib import Path
from pypersistable import TypeRegistry

TypeRegistry().register("Dog", Dog, root=Path("Animals"))
```

`register` also accepts an `importlib.metadata.EntryPoint`, which is only loaded when the type is needed.
Registering a name again raises a `RegistrationError`, unless `replace=True`, e.g. for a class redefined in a notebook.
While we could have avoided a registry, the alternative is persisting implementation-specific info,
such as the module and class names, making code refactoring very difficult.
The optional `root` groups types, e.g., you might specify a root of `Animals` for classes `Dog`, `Cat`, `Bunyip` etc.
See [DiscriminatedUnion](#discriminatedunion) for another use of root.

### `Persistable`
`Persistable` is the abstract base class for persistable objects. It provides an `object_id`, a `PersistableId`,
which is the type's registered name and a hash of the object's id fields, e.g. `Dog/3f2a...`.
Where an object's id fields contain other `Persistable` instances, only their `object_id` contribute to the parent id.

There are two species of `Persistable`:

- `Unique` - As the name suggests, this is the base for types where all fields form the id. E.g.,
```python
@dataclass
class Address(Unique):
    street_number: str  # Could be 221B, for example
    street_name: str
    town: str
    postcode: str
    country: Country
```
- `Identifiable` - A base class for objects where a subset of fields form their id. Such fields
are declared using `IdField`. E.g.,
```python
@dataclass
class Customer(Identifiable):
    name: IdField[str]
    address: Address
```

Either can equally be a pydantic `BaseModel`. `InitVar` is not allowed in id fields (or anywhere in a `Unique`),
as they are not persisted: a `TypeError` is raised when the class is defined.

### `Reference`
In classes which have user-defined types as attributes, do we want those nested objects to be persisted
by value in the parent object's payload or saved separately, with the parent persisting a reference?
The answer is very much, "it depends". If you definitely want the nested payload to be embedded, that is
easy: simply declare your type like `Customer` above. However, you could instead write it as:

```python
@dataclass
class Customer(Identifiable):
    name: IdField[str]
    address: Reference[Address]
```

This means that if we had multiple customers (e.g. family members) living at the same address,
when persisting those customers, the `Address` would be persisted once and shared, with its `object_id`
saved in the `Customer` payload.

A `Reference` may be a union of `Persistable` types, e.g. `Reference[Dog | Cat]`, as the `object_id` identifies
the type. `Reference` may also be optional or nested in containers, e.g. `tuple[Reference[Address], ...]`.
A type may reference itself, e.g. `next: "Reference[Node] | None"`, but objects may not form a reference cycle:
writing one raises a `ValueError`, as it could not be read back.

When writing as JSON, you can force references to be serialised by value via the `by_reference` parameter of
`Persistable.write`. Reading always resolves persisted references, loading them in one read per level of nesting.

### `DiscriminatedUnion`
>`DiscriminatedUnion` works on all types, they do not need to be Persistable

Polymorphism in persisted classes tends to work quite badly and should be avoided. Persistable objects
often also get used in contexts where we need to know a schema, such as web services.
`fastapi` based services, using pydantic models is a good example.

A better approach is to declare the valid types using a python `Union`. pydantic can try and infer the
actual type on deserialisation but it can be preferable to indicate that explicitly. I got tired of trying to use
half-baked implementations from Stack Overflow so I wrote `DiscriminatedUnion`, which does this by:
- Adding a type specifier to the JSON schema
- Adding that type specifier to the serialisation payload (the type's module and qualified name - no need for the
  type registry as the union tells us the type)
- Extracting and removing the type specifier from the deserialisation payload and selecting the appropriate type

```python
@dataclass
class Kennel(Identifiable):
    name: IdField[str]
    pet: DiscriminatedUnion[Dog, Cat]
```

`DiscriminatedUnion` can be combined with `__subclasses__` to implement a rather hacky polymorphism, with two
major caveats:
1. Static type checkers like pyright do not like it as the type is determined at runtime
2. You need to ensure that all subclasses of your base have been imported

For `Persistable` types `TypeRegistry.types_for_root` can help with 2., e.g.:

```python
Animals = DiscriminatedUnion[TypeRegistry().types_for_root(Path("Animals"))]
```

## Storing persistable objects

`ObjectStore` is an abstract base class for implementing an object store, such as a database.
A file system-based store, `FSStore`, is provided. Objects are written into directories, formed
from the store's root, the `TypeRegistry` root and registered name.
Each write creates a new file per type, named by the timestamp of the write, which corresponds to
`entry_time` in the persisted record's metadata, and a unique suffix, as concurrent writers may share a timestamp.

Reading uses `duckdb` to read the stored Parquet/JSON files. Currently only querying by an object's id
is supported, but this approach also allows for querying by attributes of the object in future.

A nightly job to compact the timestamped files into a single file will improve read performance.

`FSStore` uses [fsspec](https://pypi.org/project/fsspec/), which supports a wide range of file systems.

### Writing

```python
from pypersistable import FSStore, RecordFormat

# Parquet is the default
store = FSStore("file:///users/hades/ObjectStore", record_format=RecordFormat.Parquet)
best_boy = Dog(name="Cerberus")
best_boy.write(store)
```

Will result in:

```
/users/hades/ObjectStore                               # FSStore root
    /Animals                                           # TypeRegistry root
        /Dog                                           # TypeRegistry registered_name
            /2026-11-08T13~01~55.249821+00~00-3f9c1a2b.parquet  # Entry time (: is not valid on all file systems) and a unique suffix
```

If you save an object which contains references, multiple objects may be written:

```python
blofeld = Customer(
    name="Blofeld, Ernst S.",
    address=Address(
        street_number="1",
        street_name="The Hollowed Out Volcano",
        town="",
        postcode="HV1 1ZZ",
        country=Country.Krakatoa,
    ),
)
blofeld.write(store)
```

This would create:

```
/users/hades/ObjectStore                           # FSStore root
    /Customer                                      # TypeRegistry registered_name
        /2026-11-08T13~01~55.249821+00~00-3f9c1a2b.parquet  # File name
    /Address                                       # TypeRegistry registered_name
        /2026-11-08T13~01~55.249821+00~00-3f9c1a2b.parquet  # File name
```

#### Overwriting
Each object read or written records the entry time of its persisted version, as `object_entry_time`
(`None` for an object constructed anew, or by `dataclasses.replace`). `write` checks the objects written, and those
they reference, against their persisted versions:

- Unchanged objects are not written again
- An object which is out of date, i.e. persisted since its version (e.g. by someone else), raises a
  `WriteConflictError`
- If the store requires it (`require_load_of_existing=True`), an object which is persisted but was not read or written,
  e.g. one constructed from its id fields, raises a `WriteConflictError`, rather than overwriting the persisted version

```python
store = FSStore("s3://prod/ObjectStore", require_load_of_existing=True)
```

`write(..., force_overwrite=True)` asks the store to write such objects anyway. How a store treats this is up to it:
`FSStore` warns and writes them.

### Reading
Read an object by its type and id fields with `Persistable.read`:

```python
best_boy = Dog.read(store, name="Cerberus")
address = Address.read(store, street_number="1", ...)  # A Unique's id fields are all its fields
```

`UnknownObjectError` (a `KeyError`) is raised if there is no such object.
Unknown or missing id fields raise a pydantic `ValidationError`.

To read several objects, possibly of different types, in one go, pass `is_async=True` to get coroutines and await them
together. Reads from the same store (as of the same times) are batched into one `ObjectStore.read`:

```python
best_boy, blofeld = store.read_all(
    Dog.read(store, name="Cerberus", is_async=True),
    Customer.read(store, name="Blofeld, Ernst S.", is_async=True),
)
```

`read_all` runs them with `asyncio.run`, or on the `loop` you supply. Within your own async code, use e.g.
`asyncio.gather`.

`ObjectStore.read` reads by `PersistableId`, returning a dict of the objects found:

```python
loaded = store.read([best_boy.object_id, blofeld.object_id])
```

### Deleting
`Persistable.del_` (or `ObjectStore.delete`) marks objects deleted, as of an optional `effective_time`, by writing
a new version of their persisted data with `active=False`. Reads no longer find them, but their data is kept: reads
as of earlier times still do, and references to them still resolve. Writing a deleted object again restores it,
as does writing an object which references it.

```python
best_boy.del_(store)
Dog.read(store, name="Cerberus")  # Raises UnknownObjectError
best_boy.write(store)  # Restored
# Reads it, then deletes it: raises UnknownObjectError if not found
Dog.delete(store, name="Cerberus")
```

Deleting is checked for conflicts as writing is, and objects not persisted, or already deleted, are ignored.

### Bitemporality
Each record has an `effective_time`, when it takes effect, and an `entry_time`, when it was written. Pass
`effective_time` to `write` (it defaults to the entry time), and `effective_time` and/or `entry_time` to the read
methods to read the latest version as of those times:

```python
# As we knew it on 1 January
Dog.read(store, name="Cerberus", entry_time=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
```

An object read as of an earlier `entry_time` is out of date if it has since been rewritten, so writing it back
raises a `WriteConflictError` (see [Overwriting](#overwriting)).

An object can also specify an `entry_time` while being written. This allows you to update an object as of a a given
time but still allow you to see what would have been observed at the original time of writing, for example, if you
wish to back-date a change to an order.

### Schema evolution
Fields may be added to a type, with defaults, which apply when reading objects written before they were added.
Changing a field's type is not supported, nor is removing (or renaming) a field of a type which forbids extra fields
(pydantic's `extra="forbid"`): its persisted objects could no longer be read. Otherwise, a removed field's persisted
data is ignored when read (for Parquet, with a warning, once per file).

`FSStore` checks this on its first write of a type, if the type's JSON schema differs from the latest recorded for it
(in a `.jsonschema` file, named like the data files, in the type's directory). It reads, with the type, the latest
object written under each recorded schema (and any written before the first), and for Parquet compares the type's
schema with that of each persisted file. It raises an `IncompatibleSchemaError` if any is incompatible, else records
the new schema.

As it reads one object per schema, it may miss incompatibilities in fields which happen to be empty (e.g. `None`) in
that object. Validation of JSON is lax, so some type changes, e.g. from `int` to `float`, are compatible.

### JSON
Records are written as a JSON array. `bytes` are base64-encoded.

### Parquet
>`ArrowConverter` (in `pypersistable.arrow`) can be used standalone: it does not depend on `Persistable` behaviour
>and can be extended with user-defined custom behaviour. `persistable_arrow_converter()` returns one extended to handle
>`Reference` and `DiscriminatedUnion` (see [Custom types](#custom-types))

The Parquet schema is generated from the type hints, with:

- Plain `datetime`s stored as UTC timestamps, keeping their instant: writing a naive value raises a `ValueError`.
  Use pydantic's `NaiveDatetime` for a timestamp without a time zone (or `AwareDatetime` to validate as aware)
- `Decimal`s as `decimal128`, with the precision and scale from pydantic's `max_digits`/`decimal_places`
  if given, else (38, 18)
- `Enum` and `Literal` stored as their values, which must be of one type
- `tuple`, `set` and `frozenset` as lists, whose items must be of one type
- `dict` as maps
- `UUID` as strings
- `Reference` as the referenced object's `object_id`, so `by_reference=False` is not supported
- `DiscriminatedUnion` as a struct of the type specifier and a nullable field per member

Parquet has no union type: other than `T | None`, unions must be a `Reference` (of `Persistable`) or a
`DiscriminatedUnion`. Unsupported types raise `UnsupportedTypeError`.

#### Custom types
- `add_type(typ, arrow_type, to_arrow=None)`: store `typ` (and its subclasses) as `arrow_type`, with `to_arrow`
  converting values pyarrow does not accept
- `add_struct(fields)`: `fields(typ)` returns the `{name: type hint}` of the struct-like types it handles, else `None`
  (e.g. for attrs classes or msgspec Structs)
- `add_handler(handler)`: `handler(converter, hint, metadata)` returns the Arrow type for the type hints (and their
  `Annotated` metadata) it handles, else `None`

Those added most recently take precedence. `persistable_arrow_converter()` returns one with the handling for
`Reference` and `DiscriminatedUnion`, to extend and pass to `FSStore`:

```python
from ipaddress import IPv4Address

import pyarrow as pa
from pypersistable import FSStore, persistable_arrow_converter

converter = persistable_arrow_converter()
converter.add_type(IPv4Address, pa.string(), str)
store = FSStore("file:///users/hades/ObjectStore", arrow_converter=converter)
```

Note this covers the Parquet schema only: pydantic must also be able to validate and serialise the type.

## Validating dataclasses
>`ValidatingDataclassMeta` can be used independently of the `Persistable` framework

`ValidatingDataclassMeta` validates each value assigned to a field of a dataclass, with pydantic in strict mode: in
`__init__` (including defaults and `__post_init__`) and on any subsequent setting of fields.

```python
from dataclasses import dataclass, field

from pypersistable import ValidatingDataclassMeta


@dataclass(slots=True)
class Hound(metaclass=ValidatingDataclassMeta):
    name: str
    heads: int = 1
    tricks: list[str] = field(default_factory=list)


hound = Hound(name="Cerberus", heads=3)
hound.heads = "3"  # Raises pydantic.ValidationError: strict, so not coerced
```

- Dataclasses with or without slots, frozen dataclasses and subclasses are supported
- Field types are resolved on first assignment, so may refer to classes defined later, and `Annotated` constraints
  (e.g. `annotated_types.Gt`) apply
- The value pydantic returns post validation is the one stored: e.g. a copy of an assigned list
- Validation by the descriptors is disabled when dataclass instances are being constructed by pydantic validation,
  e.g. while reading persistable objects or calling `TypeAdapter(MyDataclass).validate_python(...)`, as pydantic has
  validated the fields already. Assignments in `__post_init__` are still validated, but instances constructed
  normally by custom pydantic validators are not
- `ClassVar` and `InitVar` fields, and fields whose class attribute is a data descriptor (e.g. a property), are not
  validated
- Without slots, a field's class attribute is the validating descriptor, not its default, and the default of an
  `init=False` field, never assigned, is not validated. Nor, once deleted (`del obj.field`), does reading a field
  fall back to its default or raise `AttributeError`: it reads as the descriptor, unless it is an `init=False` field
- Combining with another metaclass (e.g. `ABCMeta`) requires one derived from both
