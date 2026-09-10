"""API surface checks for the pacebrain package's top-level exports."""

import inspect

import pytest

import pacebrain


def test_all_lists_each_name_once():
    assert all(isinstance(name, str) for name in pacebrain.__all__)
    assert len(pacebrain.__all__) == len(set(pacebrain.__all__))


@pytest.mark.parametrize("name", pacebrain.__all__)
def test_every_exported_name_is_defined(name):
    assert hasattr(pacebrain, name), f"__all__ lists {name!r} but pacebrain does not define it"


def test_reexported_classes_and_functions_are_all_listed():
    """
    The reverse direction: a class or function imported into __init__ but left
    out of __all__ would work as pacebrain.X yet vanish from `import *`.
    """
    reexported = {
        name
        for name, obj in vars(pacebrain).items()
        if not name.startswith("_")
        and (inspect.isclass(obj) or inspect.isfunction(obj))
        and getattr(obj, "__module__", "").startswith("pacebrain.")
    }
    assert reexported <= set(pacebrain.__all__)
