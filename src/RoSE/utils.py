from typing import Callable, Optional


def to_list(
    x, ndim: int, dtype_caster: Optional[Callable] = None, allow_nested: bool = True
) -> list[int | float | None] | list[list[int | float | None]]:
    if x is None:
        out = [None] * ndim  # type: ignore
    elif isinstance(x, (int, float, str)):
        out = [x] * ndim  # type: ignore
    elif isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) == ndim:
            assert allow_nested, (
                "Nested structures are not allowed, but got a nested list/tuple. "
                "If you want to allow nested structures, set allow_nested=True."
            )
            # If the first element is a list or tuple, we assume it's nested
            # and we need to convert each element to a list
            out = [to_list(el, ndim, dtype_caster) for el in x]
        else:
            if len(x) != ndim:
                raise ValueError(f"got {len(x)} but expected {ndim}")
            out = list(x)
    else:
        raise TypeError(
            f"expected int | float, list[int | float], or "
            f"list[list[int | float]] but got {type(x)}"
        )
    if dtype_caster is not None:
        return sequence_to_dtype(out, dtype_caster)  # type: ignore
    return out  # type: ignore


def to_tuple(
    x, ndim: int, dtype_caster: Optional[Callable] = None, allow_nested: bool = True
) -> tuple[int | float | None, ...] | tuple[tuple[int | float | None, ...], ...]:
    if x is None:
        out = (None,) * ndim
    elif isinstance(x, (int, float, str)):
        out = (x,) * ndim
    elif isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], (list, tuple)) and len(x[0]) == ndim:
            assert allow_nested, (
                "Nested structures are not allowed, but got a nested list/tuple. "
                "If you want to allow nested structures, set allow_nested=True."
            )
            # If the first element is a list or tuple, we assume it's nested
            # and we need to convert each element to a tuple
            out = tuple(to_tuple(el, ndim, dtype_caster) for el in x)
        else:
            if len(x) != ndim:
                raise ValueError(f"got {len(x)} but expected {ndim}")
            out = tuple(x)
    else:
        raise TypeError(
            f"expected int | float, list[int | float], or "
            f"list[list[int | float]] but got {type(x)}"
        )
    if dtype_caster is not None:
        return sequence_to_dtype(out, dtype_caster)  # type: ignore
    return out


def sequence_to_dtype(
    x: (
        list[int | float | None]
        | list[list[int | float | None]]
        | tuple[int | float | None, ...]
        | tuple[tuple[int | float | None, ...], ...]
    ),
    dtype_caster: Callable,
) -> (
    list[int | float | None]
    | list[list[int | float | None]]
    | tuple[int | float | None, ...]
    | tuple[tuple[int | float | None, ...], ...]
):
    """Convert sequence elements to specified dtype."""
    if isinstance(x, list):
        if isinstance(x[0], list):
            # Nested structure
            return [list(map(dtype_caster, el)) for el in x]  # type: ignore
        return list(map(dtype_caster, x))
    elif isinstance(x, tuple):
        if isinstance(x[0], tuple):
            # Nested structure
            return tuple(tuple(map(dtype_caster, el)) for el in x)  # type: ignore
        return tuple(map(dtype_caster, x))
    else:
        raise TypeError(f"expected list or tuple but got {type(x)}")
