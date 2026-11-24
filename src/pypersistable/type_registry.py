from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Self, cast


class RegistrationError(Exception):
    def __init__(self, name: str, typ: type):
        super().__init__(f"{typ!r} already registered as {name}")


class UnregisteredTypeError(Exception):
    def __init__(self, item: str | type | Path):
        super().__init__(f"Unregistered type {item!r}")


def _registered[T](typ: type[T]) -> type[T]:
    # Tells a type that it is registered, if it asks to be: once loaded, if registered as an EntryPoint
    if (registered := getattr(typ, "_registered", None)) is not None:
        registered()
    return typ


@dataclass(frozen=True)
class TypeInfo[T]:
    name: str
    root: Path
    typ: type[T] | EntryPoint

    @property
    def resolved_type(self) -> type[T]:
        if not isinstance(self.typ, EntryPoint):
            return self.typ
        return _registered(cast(type[T], self.typ.load()))


class TypeRegistry[T]:
    __instance: Self | None = None
    __initialised: bool = False

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)

        return cls.__instance

    def __init__(self):
        if not self.__initialised:
            self.__by_name: dict[str, TypeInfo[T]] = {}
            self.__by_type: dict[type, TypeInfo[T]] = {}
            self.__by_type_name: dict[str, TypeInfo[T]] = {}
            self.__by_root: dict[Path, list[TypeInfo[T]]] = {}
            self.__initialised = True

    def __type_info(self, typ: type[T] | EntryPoint) -> TypeInfo[T]:
        try:
            if isinstance(typ, EntryPoint):
                return self.__by_type_name[typ.value]
            else:
                try:
                    return self.__by_type[typ]
                except KeyError:
                    return self.__by_type_name[f"{typ.__module__}:{typ.__qualname__}"]
        except KeyError:
            raise UnregisteredTypeError(
                typ.value if isinstance(typ, EntryPoint) else typ
            )

    def registered_name(self, typ: type[T]) -> str:
        return self.__type_info(typ).name

    def root(self, typ: type[T]) -> Path:
        return self.__type_info(typ).root

    def get_type(self, name: str) -> type[T]:
        try:
            return self.__by_name[name].resolved_type
        except KeyError:
            raise UnregisteredTypeError(name)

    def types_for_root(self, root: Path) -> tuple[type[T], ...]:
        try:
            return tuple(t.resolved_type for t in self.__by_root[root])
        except KeyError:
            raise UnregisteredTypeError(root)

    def register(
        self, name: str, typ: type[T] | EntryPoint, root=Path(), replace: bool = False
    ):
        """replace: replace the type registered as name, if any, e.g. with a redefinition of it"""
        type_info = self.__by_name.get(name)
        if type_info is not None:
            if not replace:
                raise RegistrationError(name, type_info.resolved_type)

            del self.__by_name[name]
            self.__by_root[type_info.root].remove(type_info)
            if isinstance(type_info.typ, EntryPoint):
                del self.__by_type_name[type_info.typ.value]
            else:
                del self.__by_type[type_info.typ]

        try:
            type_info = self.__type_info(typ)
        except UnregisteredTypeError:
            pass
        else:
            raise RegistrationError(type_info.name, type_info.resolved_type)

        type_info = TypeInfo(name, root, typ)
        self.__by_name[name] = type_info
        self.__by_root.setdefault(root, []).append(type_info)

        if isinstance(typ, EntryPoint):
            self.__by_type_name[typ.value] = type_info
        else:
            self.__by_type[_registered(typ)] = type_info
