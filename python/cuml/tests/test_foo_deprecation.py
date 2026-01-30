#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sys
import warnings

import pytest


def _clear_foo_modules():
    """Clear foo-related modules from sys.modules to allow re-import."""
    to_remove = [key for key in sys.modules if "foo" in key]
    for key in to_remove:
        del sys.modules[key]


def test_from_cuml_import_foo_warns():
    """from cuml import foo should warn."""
    _clear_foo_modules()
    with pytest.warns(FutureWarning, match="cuml.foo.*deprecated"):
        from cuml import foo

    assert hasattr(foo, "ExampleClass")


def test_from_cuml_foo_import_bar_warns():
    """from cuml.foo import ExampleClass should warn."""
    _clear_foo_modules()
    with pytest.warns(FutureWarning, match="cuml.foo.*deprecated"):
        from cuml.foo import ExampleClass

    assert ExampleClass is not None


def test_import_cuml_foo_warns():
    """import cuml.foo should warn."""
    _clear_foo_modules()
    with pytest.warns(FutureWarning, match="cuml.foo.*deprecated"):
        import cuml.foo

    assert hasattr(cuml.foo, "ExampleClass")


def test_from_cuml_private_foo_no_warning():
    """from cuml._foo import ExampleClass should NOT warn."""
    _clear_foo_modules()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        from cuml._foo import ExampleClass

    assert ExampleClass is not None


def test_import_cuml_private_foo_no_warning():
    """import cuml._foo should NOT warn."""
    _clear_foo_modules()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        import cuml._foo

    assert hasattr(cuml._foo, "ExampleClass")


def test_functionality_preserved():
    """Ensure functionality works through both paths."""
    _clear_foo_modules()
    from cuml._foo import ExampleClass, example_function

    obj = ExampleClass(42)
    assert obj.value == 42
    assert example_function(5) == 10
