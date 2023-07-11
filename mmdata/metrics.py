# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define metrics compatible with DataFrame.
"""

# Imports
import mock
import numpy as np
from types import MethodType
from functools import partial
from sklearn.metrics._scorer import _PredictScorer


def make_scorer(score_func, *, y_col=None, greater_is_better=True,
                needs_proba=False, needs_threshold=False, **kwargs):
    """ Make a scorer from a performance metric or loss function.
    For more info see `sklearn.metrics.make_scorer`.

    Parameters
    ----------
    score_func: callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    greater_is_better: bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.
    needs_proba: bool, default=False
        Whether `score_func` requires `predict_proba` to get probability
        estimates out of a classifier.
        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).
    needs_threshold: bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.
        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).
        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.
    **kwargs: additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer: callable
        Callable object that returns a scalar score; greater is better.

    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function` or :term:`predict_proba` when
    :term:`decision_function` is not present.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )
    if needs_proba:
        raise NotImplementedError
    elif needs_threshold:
        raise NotImplementedError
    else:
        cls = _PredictScorerDF
    return cls(score_func, sign, y_col, kwargs)


class _PredictScorerDF(_PredictScorer):
    def __init__(self, score_func, sign, y_col, kwargs):
        self.y_col = y_col
        super().__init__(score_func, sign, kwargs)

    def _score(self, method_caller, estimator, X, y_true, **kwargs):
        """ Evaluate predicted target values for X relative to y_true.
        """
        self._warn_overlap(
            message=(
                "There is an overlap between set kwargs of this scorer "
                "instance and passed metadata. Please pass them either as "
                "kwargs to `make_scorer` or metadata, but not both."
            ),
            kwargs=kwargs,
        )
        method_caller = partial(_cached_call_df, None)
        y_pred = method_caller(estimator, "predict", X, y_true)
        scoring_kwargs = {**self._kwargs, **kwargs}
        return self._sign * self._score_func(y_true[self.y_col].values,
                                             y_pred, **scoring_kwargs)


def _cached_call_df(cache, estimator, response_method, *args, **kwargs):
    """ Call estimator with method and args and kwargs.
    """
    if cache is not None and response_method in cache:
        return cache[response_method]
    result, _ = _get_response_values_df(
        estimator, *args, response_method=response_method, **kwargs
    )
    if cache is not None:
        cache[response_method] = result
    return result


def _get_response_values_df(estimator, X, y, response_method, pos_label=None,):
    """ Compute the response values of a classifier or a regressor.

    The response values are predictions, one scalar value for each sample in X
    that depends on the specific choice of `response_method`.

    If `estimator` is a binary classifier, also return the label for the
    effective positive class.

    .. versionadded:: 1.3

    Parameters
    ----------
    estimator: estimator instance
        Fitted classifier or regressor or a fitted
        :class:`~sklearn.pipeline.Pipeline` in which the last estimator is a
        classifier or a regressor.
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y: DataFrame (n_samples, n_lables)
        the variables.
    response_method: {"predict_proba", "decision_function", "predict"} or \
            list of such str
        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`decision_function` or
        :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.
    pos_label: int, float, bool or str, default=None
        The class considered as the positive class when computing
        the metrics. By default, `estimators.classes_[1]` is
        considered as the positive class.

    Returns
    -------
    y_pred: ndarray of shape (n_samples,)
        Target scores calculated from the provided response_method
        and `pos_label`.
    pos_label: int, float, bool, str or None
        The class considered as the positive class when computing
        the metrics. Returns `None` if `estimator` is a regressor.

    Raises
    ------
    ValueError
        If `pos_label` is not a valid label.
        If the shape of `y_pred` is not consistent for binary classifier.
        If the response method can be applied to a classifier only and
        `estimator` is a regressor.
    """
    from sklearn.base import is_classifier  # noqa
    from sklearn.utils.validation import _check_response_method  # noqa

    if is_classifier(estimator):
        prediction_method = _check_response_method(estimator, response_method)
        classes = estimator.classes_
        target_type = "binary" if len(classes) <= 2 else "multiclass"

        if pos_label is not None and pos_label not in classes.tolist():
            raise ValueError(
                f"pos_label={pos_label} is not a valid label: It should be "
                f"one of {classes}"
            )
        elif pos_label is None and target_type == "binary":
            pos_label = pos_label if pos_label is not None else classes[-1]

        y_pred = prediction_method(X, y)
        if prediction_method.__name__ == "predict_proba":
            if target_type == "binary" and y_pred.shape[1] <= 2:
                if y_pred.shape[1] == 2:
                    col_idx = np.flatnonzero(classes == pos_label)[0]
                    y_pred = y_pred[:, col_idx]
                else:
                    err_msg = (
                        f"Got predict_proba of shape {y_pred.shape}, but need "
                        "classifier with two classes."
                    )
                    raise ValueError(err_msg)
        elif prediction_method.__name__ == "decision_function":
            if target_type == "binary":
                if pos_label == classes[0]:
                    y_pred *= -1
    else:  # estimator is a regressor
        if response_method != "predict":
            raise ValueError(
                f"{estimator.__class__.__name__} should either be a "
                "classifier to be used with response_method="
                f"{response_method} or the response_method "
                "should be 'predict'. Got a regressor with response_method="
                f"{response_method} instead."
            )
        y_pred, pos_label = estimator.predict(X, y), None

    return y_pred, pos_label
