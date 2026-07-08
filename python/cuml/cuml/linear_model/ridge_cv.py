#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.internals.outputs import reflect
from cuml.internals.validation import check_inputs
from cuml.linear_model.base import LinearPredictMixin
from cuml.linear_model.ridge import Ridge

# The GCV decomposition is computed in the input dtype (float32 stays float32).
# scikit-learn always upcasts to float64 for numerical robustness, so float32
# results may differ from sklearn by more than float64 results do. This dtype is
# only used for the cheap scalar scoring in the explicit-``cv`` k-fold path,
# where float64 keeps the R^2 computation stable at negligible cost.
_SCORE_DTYPE = cp.float64


def _diag_dot(D, B):
    """Compute ``dot(diag(D), B)`` for a 1d ``D`` and 1d/2d ``B``."""
    if B.ndim > 1:
        D = D[(slice(None),) + (None,) * (B.ndim - 1)]
    return D * B


def _decomp_diag(v_prime, Q):
    """Compute the diagonal of ``dot(Q, dot(diag(v_prime), Q.T))``."""
    return cp.sum(v_prime * Q**2, axis=1)


def _check_gcv_mode(n_samples, n_features, gcv_mode):
    """Resolve the ``gcv_mode`` parameter to a concrete strategy."""
    if gcv_mode == "svd":
        return "svd"
    # "auto" and "eigen" both fall back to a Gram/covariance eigendecomposition,
    # picking whichever matrix is smaller.
    return "gram" if n_samples <= n_features else "cov"


class RidgeCV(
    InteropMixin,
    RegressorMixin,
    LinearPredictMixin,
    FMajorInputTagMixin,
    Base,
):
    """Ridge regression with built-in cross-validation.

    By default, it performs efficient Leave-One-Out Cross-Validation (via
    Generalized Cross-Validation) to select the best ``alpha`` from ``alphas``.
    When an explicit ``cv`` is provided, a k-fold search over :class:`~cuml.Ridge`
    is used instead.

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try. Regularization strength; must be a
        positive float. When using Leave-One-Out cross-validation (``cv=None``),
        alphas must be strictly positive.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations (i.e. data is expected to be
        centered).
    scoring : str, callable, default=None
        Only ``None`` (negative mean squared error for ``cv=None``, or
        :math:`R^2` for an explicit ``cv``) is supported on GPU. A non-``None``
        value raises ``NotImplementedError``.
    cv : int or None, default=None
        Determines the cross-validation splitting strategy:

        - ``None``, to use efficient Leave-One-Out cross-validation.
        - integer, to specify the number of folds.

    gcv_mode : {'auto', 'svd', 'eigen'}, default=None
        Flag indicating which strategy to use when performing Leave-One-Out
        cross-validation:

        - ``'auto'`` / ``None`` : same as ``'eigen'``.
        - ``'svd'`` : use a singular value decomposition of X.
        - ``'eigen'`` : use an eigendecomposition of ``X X'`` when
          ``n_samples <= n_features`` or ``X' X`` otherwise.

    store_cv_results : bool, default=False
        Whether to store the cross-validation values in the ``cv_results_``
        attribute. Only compatible with ``cv=None``.
    alpha_per_target : bool, default=False
        Whether to optimize the alpha value separately for each target (for
        multi-output settings). Only compatible with ``cv=None``.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    cv_results_ : array of shape (n_samples, n_alphas) or \
            (n_samples, n_targets, n_alphas), optional
        Cross-validation values for each alpha (only available if
        ``store_cv_results=True`` and ``cv=None``).
    coef_ : array of shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    intercept_ : float or array of shape (n_targets,)
        Independent term in the decision function. Set to 0.0 if
        ``fit_intercept=False``.
    alpha_ : float or array of shape (n_targets,)
        Estimated regularization parameter, or, if ``alpha_per_target=True``,
        the estimated regularization parameter for each target.
    best_score_ : float or array of shape (n_targets,)
        Score of the base estimator with the best alpha.

    Notes
    -----
    For additional docs, see `scikit-learn's RidgeCV
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>`_.

    Examples
    --------
    >>> from cuml.datasets import make_regression
    >>> from cuml.linear_model import RidgeCV
    >>> X, y = make_regression(n_samples=50, n_features=5, random_state=0)
    >>> reg = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)
    >>> reg.alpha_  # doctest: +SKIP
    0.1
    """

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()
    cv_results_ = CumlArrayDescriptor()

    _cpu_class_path = "sklearn.linear_model.RidgeCV"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "alphas",
            "fit_intercept",
            "scoring",
            "cv",
            "gcv_mode",
            "store_cv_results",
            "alpha_per_target",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.scoring is not None:
            raise UnsupportedOnGPU("Custom `scoring` is not supported")

        if model.cv is not None:
            # The k-fold path relies on matching scikit-learn's fold splitting
            # and scoring exactly, which we don't attempt. Fall back to CPU.
            raise UnsupportedOnGPU("`cv != None` is not supported")

        return {
            "alphas": model.alphas,
            "fit_intercept": model.fit_intercept,
            "scoring": model.scoring,
            "cv": model.cv,
            "gcv_mode": model.gcv_mode,
            "store_cv_results": model.store_cv_results,
            "alpha_per_target": model.alpha_per_target,
        }

    def _params_to_cpu(self):
        return {
            "alphas": self.alphas,
            "fit_intercept": self.fit_intercept,
            "scoring": self.scoring,
            "cv": self.cv,
            "gcv_mode": self.gcv_mode,
            "store_cv_results": self.store_cv_results,
            "alpha_per_target": self.alpha_per_target,
        }

    def _attrs_from_cpu(self, model):
        out = {
            "coef_": to_gpu(model.coef_),
            "intercept_": to_gpu(model.intercept_),
            "alpha_": model.alpha_,
            "best_score_": model.best_score_,
            **super()._attrs_from_cpu(model),
        }
        if self.store_cv_results and hasattr(model, "cv_results_"):
            out["cv_results_"] = to_gpu(model.cv_results_)
        return out

    def _attrs_to_cpu(self, model):
        out = {
            "coef_": to_cpu(self.coef_),
            "intercept_": to_cpu(self.intercept_),
            "alpha_": self.alpha_,
            "best_score_": self.best_score_,
            **super()._attrs_to_cpu(model),
        }
        if self.store_cv_results and hasattr(self, "cv_results_"):
            out["cv_results_"] = to_cpu(self.cv_results_)
        return out

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        gcv_mode=None,
        store_cv_results=False,
        alpha_per_target=False,
        output_type=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.alpha_per_target = alpha_per_target

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    # ------------------------------------------------------------------
    # GCV decomposition/solve helpers (dense-only ports of sklearn's
    # ``_RidgeGCV``). All computation happens in float64.
    # ------------------------------------------------------------------
    def _solve_eigen_gram(
        self, alpha, y, sqrt_sw, eigvals, Q, QT_y, QT_sqrt_sw, XT
    ):
        w = 1.0 / (eigvals + alpha)
        c = Q @ _diag_dot(w, QT_y)
        d = _decomp_diag(w, Q)
        if self.fit_intercept:
            sw_sum = sqrt_sw @ sqrt_sw
            Ginv_sqrt_sw = Q @ _diag_dot(w, QT_sqrt_sw)
            d -= Ginv_sqrt_sw * sqrt_sw / sw_sum
        if y.ndim == 2:
            d = d[:, None]
        looe = c / d
        coef = XT @ c
        return looe, coef

    def _solve_eigen_covariance(
        self, alpha, y, sqrt_sw, eigvals, V, X, XT_y, XT_sqrt_sw
    ):
        w = 1.0 / (eigvals + alpha)
        Hinv = (V * w) @ V.T
        Hinv_XT_y = Hinv @ XT_y
        Hinv_XT_sqrt_sw = Hinv @ XT_sqrt_sw
        X_Hinv_XT_y = X @ Hinv_XT_y
        X_Hinv_XT_sqrt_sw = X @ Hinv_XT_sqrt_sw
        alpha_c = y - X_Hinv_XT_y
        alpha_d = 1 - cp.sum((X @ Hinv) * X, axis=1)
        if self.fit_intercept:
            sw_sum = sqrt_sw @ sqrt_sw
            alpha_Ginv_sqrt_sw = sqrt_sw - X_Hinv_XT_sqrt_sw
            alpha_d -= alpha_Ginv_sqrt_sw * sqrt_sw / sw_sum
        if y.ndim == 2:
            alpha_d = alpha_d[:, None]
        looe = alpha_c / alpha_d
        coef = Hinv_XT_y
        return looe, coef

    def _solve_svd_design_matrix(
        self, alpha, y, sqrt_sw, singvals, U, V, UT_y, UT_sqrt_sw
    ):
        n_samples, n_features = U.shape[0], V.shape[0]
        if n_samples <= n_features:
            # Wide X case (n_samples <= n_features).
            w = alpha / (singvals**2 + alpha)
            alpha_c = U @ _diag_dot(w, UT_y)
            alpha_d = _decomp_diag(w, U)
        else:
            # Long X case (n_features < n_samples).
            w = alpha / (singvals**2 + alpha) - 1
            alpha_c = U @ _diag_dot(w, UT_y) + y
            alpha_d = _decomp_diag(w, U) + 1
        if self.fit_intercept:
            sw_sum = sqrt_sw @ sqrt_sw
            if n_samples <= n_features:
                alpha_Ginv_sqrt_sw = U @ _diag_dot(w, UT_sqrt_sw)
            else:
                alpha_Ginv_sqrt_sw = U @ _diag_dot(w, UT_sqrt_sw) + sqrt_sw
            alpha_d -= alpha_Ginv_sqrt_sw * sqrt_sw / sw_sum
        if y.ndim == 2:
            alpha_d = alpha_d[:, None]
        looe = alpha_c / alpha_d
        coef = V @ _diag_dot(singvals / (singvals**2 + alpha), UT_y)
        return looe, coef

    def _fit_gcv(self, X, y, sample_weight, input_dtype):
        """Fit via Generalized (leave-one-out) cross-validation."""
        n_samples, n_features = X.shape
        # Compute in the input dtype (float32 stays float32).
        work = X.dtype

        alphas = np.asarray(self.alphas, dtype=np.float64)
        if alphas.ndim == 0:
            alphas = alphas.reshape(1)
        if (alphas <= 0).any():
            raise ValueError(
                "alphas must be strictly positive when cv=None, got "
                f"{self.alphas}"
            )

        # Center and rescale following sklearn's `_preprocess_data`
        # (`rescale_with_sw=True`).
        if self.fit_intercept:
            if sample_weight is not None:
                sw_sum = sample_weight.sum()
                X_offset = (X * sample_weight[:, None]).sum(axis=0) / sw_sum
                y_offset = (
                    (y * sample_weight[:, None]).sum(axis=0) / sw_sum
                    if y.ndim == 2
                    else (y * sample_weight).sum() / sw_sum
                )
            else:
                X_offset = X.mean(axis=0)
                y_offset = y.mean(axis=0)
            X = X - X_offset
            y = y - y_offset
        else:
            X_offset = cp.zeros(n_features, dtype=work)
            y_offset = (
                cp.zeros(y.shape[1], dtype=work)
                if y.ndim == 2
                else work.type(0.0)
            )

        if sample_weight is not None:
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * (sqrt_sw[:, None] if y.ndim == 2 else sqrt_sw)
        else:
            sqrt_sw = cp.ones(n_samples, dtype=work)

        gcv_mode = _check_gcv_mode(
            n_samples, n_features, self.gcv_mode or "auto"
        )
        if gcv_mode == "gram":
            K = X @ X.T
            eigvals, Q = cp.linalg.eigh(K)
            decomposition = (eigvals, Q, Q.T @ y, Q.T @ sqrt_sw, X.T)
            solve = self._solve_eigen_gram
        elif gcv_mode == "cov":
            cov = X.T @ X
            eigvals, V = cp.linalg.eigh(cov)
            decomposition = (eigvals, V, X, X.T @ y, X.T @ sqrt_sw)
            solve = self._solve_eigen_covariance
        else:
            U, singvals, VT = cp.linalg.svd(X, full_matrices=False)
            decomposition = (singvals, U, VT.T, U.T @ y, U.T @ sqrt_sw)
            solve = self._solve_svd_design_matrix

        n_y = 1 if y.ndim == 1 else y.shape[1]
        n_alphas = len(alphas)
        per_target = self.alpha_per_target and n_y > 1

        if self.store_cv_results:
            cv_results = cp.empty((n_samples * n_y, n_alphas), dtype=work)

        best_coef = best_score = best_alpha = None
        for i, alpha in enumerate(alphas):
            looe, coef = solve(float(alpha), y, sqrt_sw, *decomposition)
            squared_errors = looe**2
            if self.store_cv_results:
                cv_results[:, i] = squared_errors.reshape(-1)

            if per_target:
                score = cp.mean(-squared_errors, axis=0)
            else:
                score = cp.mean(-squared_errors)

            if best_score is None:
                best_coef = coef
                if per_target:
                    best_score = score.reshape(-1)
                    best_alpha = cp.full(n_y, alpha, dtype=work)
                else:
                    best_score = score
                    best_alpha = alpha
            elif per_target:
                to_update = score > best_score
                best_coef[:, to_update] = coef[:, to_update]
                best_score[to_update] = score[to_update]
                best_alpha[to_update] = alpha
            elif score > best_score:
                best_coef, best_score, best_alpha = coef, score, alpha

        coef = best_coef
        if y.ndim == 2:
            coef = coef.T
        if y.ndim == 1 or y.shape[1] == 1:
            coef = coef.reshape(-1)

        # Set intercept, mirroring `LinearModel._set_intercept`.
        if self.fit_intercept:
            if coef.ndim == 1:
                intercept = y_offset - X_offset @ coef
            else:
                intercept = y_offset - X_offset @ coef.T
        else:
            intercept = 0.0

        # Cast fitted attributes back to the input dtype (like sklearn).
        self.coef_ = CumlArray(coef.astype(input_dtype))
        if cp.isscalar(intercept) or getattr(intercept, "ndim", 1) == 0:
            self.intercept_ = float(intercept)
        else:
            self.intercept_ = CumlArray(
                cp.asarray(intercept, dtype=input_dtype)
            )
        # `alpha_`/`best_score_` are small metadata; keep them on host to match
        # scikit-learn (a float, or a numpy array when alpha_per_target=True).
        if per_target:
            self.alpha_ = cp.asnumpy(cp.asarray(best_alpha)).astype(
                input_dtype
            )
            self.best_score_ = cp.asnumpy(cp.asarray(best_score)).astype(
                input_dtype
            )
        else:
            self.alpha_ = float(best_alpha)
            self.best_score_ = float(best_score)

        if self.store_cv_results:
            if y.ndim == 1:
                shape = (n_samples, n_alphas)
            else:
                shape = (n_samples, n_y, n_alphas)
            self.cv_results_ = CumlArray(
                cv_results.reshape(shape).astype(input_dtype)
            )

    def _fit_cv(self, X, y, sample_weight, input_dtype):
        """Fit via an explicit k-fold search over :class:`~cuml.Ridge`."""
        from cuml.model_selection import KFold

        if self.store_cv_results:
            raise ValueError(
                "cv!=None and store_cv_results=True are incompatible"
            )
        if self.alpha_per_target:
            raise ValueError(
                "cv!=None and alpha_per_target=True are incompatible"
            )

        alphas = np.asarray(self.alphas, dtype=np.float64)
        if alphas.ndim == 0:
            alphas = alphas.reshape(1)
        if (alphas < 0).any():
            raise ValueError(f"alphas must be non-negative, got {self.alphas}")

        if isinstance(self.cv, int):
            splitter = KFold(n_splits=self.cv)
            splits = list(splitter.split(X, y))
        elif hasattr(self.cv, "split"):
            splits = list(self.cv.split(X, y))
        else:
            splits = list(self.cv)

        mean_scores = cp.empty(len(alphas), dtype=_SCORE_DTYPE)
        for i, alpha in enumerate(alphas):
            fold_scores = []
            for train, test in splits:
                train = cp.asarray(train)
                test = cp.asarray(test)
                model = Ridge(
                    alpha=float(alpha),
                    fit_intercept=self.fit_intercept,
                    output_type="cupy",
                )
                if sample_weight is None:
                    model.fit(X[train], y[train])
                else:
                    model.fit(
                        X[train], y[train], sample_weight=sample_weight[train]
                    )
                y_pred = model.predict(X[test])
                fold_scores.append(_r2_score(y[test], y_pred))
            mean_scores[i] = sum(fold_scores) / len(fold_scores)

        best_idx = int(cp.argmax(mean_scores))
        best_alpha = float(alphas[best_idx])

        model = Ridge(
            alpha=best_alpha,
            fit_intercept=self.fit_intercept,
            output_type="cupy",
        )
        if sample_weight is None:
            model.fit(X, y)
        else:
            model.fit(X, y, sample_weight=sample_weight)

        self.coef_ = CumlArray(cp.asarray(model.coef_, dtype=input_dtype))
        intercept = model.intercept_
        if isinstance(intercept, CumlArray):
            intercept = intercept.to_output("cupy")
        if cp.isscalar(intercept) or getattr(intercept, "ndim", 1) == 0:
            self.intercept_ = float(intercept)
        else:
            self.intercept_ = CumlArray(
                cp.asarray(intercept, dtype=input_dtype)
            )
        self.alpha_ = best_alpha
        self.best_score_ = float(mean_scores[best_idx])

    @generate_docstring()
    @reflect(reset=True)
    def fit(self, X, y, sample_weight=None) -> "RidgeCV":
        """Fit the RidgeCV model with X and y."""
        if self.scoring is not None:
            raise NotImplementedError(
                "Only `scoring=None` is supported on GPU"
            )

        X, y, sample_weight = check_inputs(
            self,
            X,
            y,
            sample_weight,
            dtype=("float32", "float64"),
            accept_sparse=False,
            accept_multi_output=True,
            ensure_min_samples=2,
            reset=True,
        )
        input_dtype = X.dtype

        # Keep the input dtype (float32 stays float32) rather than upcasting.
        X = cp.asarray(X)
        y = cp.asarray(y)
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)

        if self.cv is None:
            self._fit_gcv(X, y, sample_weight, input_dtype)
        else:
            self._fit_cv(X, y, sample_weight, input_dtype)

        return self


def _r2_score(y_true, y_pred):
    """Uniform-average :math:`R^2` matching sklearn's default scorer."""
    y_true = cp.asarray(y_true, dtype=_SCORE_DTYPE)
    y_pred = cp.asarray(y_pred, dtype=_SCORE_DTYPE)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return float(cp.mean(r2))
