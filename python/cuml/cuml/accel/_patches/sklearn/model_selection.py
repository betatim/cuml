# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import functools
import os

import cupy as cp
import numpy as np
import scipy._lib._array_api as _scipy_array_api
import scipy.sparse
import sklearn
from packaging.version import Version
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from cuml.accel.core import logger
from cuml.accel.estimator_proxy import ensure_host, is_proxy
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.input_utils import is_array_like
from cuml.internals.outputs import using_output_type

AT_LEAST_SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0")

__all__ = ("GridSearchCV",)


@contextlib.contextmanager
def _enable_scipy_array_api():
    """Enable scipy's array API support.

    Sets the SCIPY_ARRAY_API env var (checked by sklearn's config validation)
    and updates scipy's cached config (in case scipy had already been imported).

    Both are restored on exit.
    """
    old_env = os.environ.get("SCIPY_ARRAY_API")
    os.environ["SCIPY_ARRAY_API"] = "1"

    old_cached = _scipy_array_api._GLOBAL_CONFIG["SCIPY_ARRAY_API"]
    _scipy_array_api._GLOBAL_CONFIG["SCIPY_ARRAY_API"] = "1"

    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("SCIPY_ARRAY_API", None)
        else:
            os.environ["SCIPY_ARRAY_API"] = old_env

        _scipy_array_api._GLOBAL_CONFIG["SCIPY_ARRAY_API"] = old_cached


def _has_custom_scorer(scoring):
    """Determine if scoring contains user-provided callables.

    Custom scorers expect numpy arrays, but the array API path
    uses cupy arrays. Bail out of the optimization so the user's
    code runs unchanged on CPU.
    """
    if scoring is None or isinstance(scoring, str):
        return False
    if isinstance(scoring, (list, tuple)):
        return not all(isinstance(s, str) for s in scoring)
    if isinstance(scoring, dict):
        return not all(isinstance(v, str) for v in scoring.values())
    return True


def _contains_proxy(estimator):
    """Check if an estimator can benefit from the cupy data path.

    For estimator proxies this is always True. For Pipelines, ALL steps must
    be proxies: non-proxy steps can't handle cupy inputs, and a non-proxy
    tail would produce numpy predictions causing a device mismatch with
    cupy y_test in scoring.
    """
    if is_proxy(estimator):
        return True
    if isinstance(estimator, Pipeline):
        return all(
            is_proxy(step)
            for _, step in estimator.steps
            if step not in (None, "passthrough")
        )
    return False


def _patch_fit(cls):
    orig_fit = cls.fit

    @functools.wraps(orig_fit)
    def fit(self, X, y=None, **params):
        estimator_name = type(self.estimator).__name__

        if not AT_LEAST_SKLEARN_18:
            logger.debug(
                "`GridSearchCV.fit` not optimized: requires sklearn >= 1.8"
            )
            return orig_fit(self, X, y, **params)

        if not _contains_proxy(self.estimator):
            logger.debug(
                f"`GridSearchCV.fit` not optimized: "
                f"`{estimator_name}` does not contain accelerated estimators"
            )
            return orig_fit(self, X, y, **params)

        if scipy.sparse.issparse(X):
            logger.debug("`GridSearchCV.fit` not optimized: sparse input")
            return orig_fit(self, X, y, **params)

        if np.asarray(X).dtype.kind not in "fiub" or (
            y is not None and np.asarray(y).dtype.kind not in "fiub"
        ):
            logger.debug("`GridSearchCV.fit` not optimized: non-numeric data")
            return orig_fit(self, X, y, **params)

        if self.n_jobs is not None and self.n_jobs != 1:
            logger.debug(
                f"`GridSearchCV.fit` not optimized: n_jobs={self.n_jobs} "
                f"(set n_jobs=1 for GPU acceleration)"
            )
            return orig_fit(self, X, y, **params)

        if _has_custom_scorer(self.scoring):
            logger.debug("`GridSearchCV.fit` not optimized: custom scorer")
            return orig_fit(self, X, y, **params)

        logger.debug(
            f"`GridSearchCV.fit` input data moved to GPU as some "
            f"parameter combinations support acceleration for "
            f"`{estimator_name}`"
        )

        X_gpu = cp.asarray(X) if not isinstance(X, cp.ndarray) else X
        y_gpu = (
            cp.asarray(y)
            if y is not None and not isinstance(y, cp.ndarray)
            else y
        )
        # Convert array-like params (e.g. sample_weight) to cupy so CV
        # splitting with cupy indices works. Exclude "groups" which goes to
        # the CV splitter and must stay on host.
        params = {
            k: cp.asarray(v)
            if k != "groups" and is_array_like(v, accept_lists=True)
            else v
            for k, v in params.items()
        }

        with (
            _enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_fit(self, X_gpu, y_gpu, **params)

        # Ensure user-facing attributes are host arrays
        if GlobalSettings().output_type in (None, "numpy"):
            for attr in ("best_score_", "best_index_"):
                val = getattr(self, attr, None)
                if val is not None:
                    setattr(self, attr, ensure_host(val))

        return out

    cls.fit = fit


_patch_fit(GridSearchCV)
