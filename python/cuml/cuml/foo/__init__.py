#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sys
import warnings

# Issue deprecation warning when this module is imported directly
warnings.warn(
    "The 'cuml.foo' module is deprecated and will be removed in 26.06. "
    "Please use 'cuml._foo' instead.",
    FutureWarning,
    stacklevel=2,
)

# Ensure submodule attribute access also works (import cuml.foo; cuml.foo.example)
from cuml import _foo  # noqa: E402

# Re-export everything from the private module
from cuml._foo import *  # noqa: E402, F403
from cuml._foo import __all__  # noqa: E402

sys.modules[__name__].__dict__.update(_foo.__dict__)
