#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
__all__ = [
    "create_class_prediction_error_chart",
    "create_classification_report_chart",
    "create_classifier_summary",
    "create_confusion_matrix_chart",
    "create_cooks_distance_chart",
    "create_feature_importance_chart",
    "create_kelbow_chart",
    "create_kmeans_summary",
    "create_learning_curve_chart",
    "create_precision_recall_chart",
    "create_prediction_error_chart",
    "create_regressor_summary",
    "create_residuals_chart",
    "create_roc_auc_chart",
    "create_silhouette_chart",
    "get_cluster_labels",
    "get_estimator_params",
    "get_pickled_model",
    "get_scores",
    "get_test_preds",
    "get_test_preds_proba",
]

import matplotlib.pyplot as plt
import pandas as pd
from scikitplot.estimators import plot_learning_curve
from scikitplot.metrics import plot_precision_recall
from sklearn.base import (
    BaseEstimator,
    is_classifier,
    is_regressor,
)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from yellowbrick.classifier import (
    ROCAUC,
    ClassificationReport,
    ClassPredictionError,
    ConfusionMatrix,
)
from yellowbrick.cluster import (
    KElbowVisualizer,
    SilhouetteVisualizer,
)
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.regressor import (
    CooksDistance,
    PredictionError,
    ResidualsPlot,
)

from neptune_sklearn.impl.version import __version__

try:
    from neptune.types import (
        File,
        FileSeries,
    )
    from neptune.utils import stringify_unsupported
except ImportError:
    from neptune.new.types import (
        File,
        FileSeries,
    )
    from neptune.new.utils import stringify_unsupported


def create_regressor_summary(regressor, X_train, X_test, y_train, y_test, nrows=1000, log_charts=True):
    """Creates scikit-learn regressor summary.

    The summary includes:

    - all regressor parameters,
    - pickled estimator (model),
    - test predictions,
    - test scores,
    - model performance visualizations.

    The regressor should be fitted before calling this function.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The regression target for training.
        y_test (`ndarray`): The regression target for testing.
        nrows (`int`, optional): Log first `nrows` rows of test predictions.
        log_charts (`bool`, optional): Whether to calculate and log chart visualizations.
            Note: Calculating visualizations is potentially expensive depending on input data and regressor,
            and may take some time to finish. This is equivalent to calling the following functions from
            this module: `create_learning_curve_chart()`, `create_feature_importance_chart()`,
            `create_residuals_chart()`, `create_prediction_error_chart()`, and `create_cooks_distance_chart()`.

    Returns:
        `dict` with all summary items.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["random_forest/summary"] = npt_utils.create_regressor_summary(rfr, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    reg_summary = dict()

    reg_summary["all_params"] = stringify_unsupported(get_estimator_params(regressor))
    reg_summary["pickled_model"] = get_pickled_model(regressor)

    y_pred = regressor.predict(X_test)

    reg_summary["test"] = {
        "preds": get_test_preds(regressor, X_test, y_test, y_pred=y_pred, nrows=nrows),
        "scores": get_scores(regressor, X_test, y_test, y_pred=y_pred),
    }

    reg_summary["integration/about/neptune-sklearn"] = __version__

    if log_charts:
        learning_curve = create_learning_curve_chart(regressor, X_train, y_train)
        feature_importance = create_feature_importance_chart(regressor, X_train, y_train)
        residuals = create_residuals_chart(regressor, X_train, X_test, y_train, y_test)
        prediction_error = create_prediction_error_chart(regressor, X_train, X_test, y_train, y_test)
        cooks_distance = create_cooks_distance_chart(regressor, X_train, y_train)

        if learning_curve:
            reg_summary["diagnostics_charts/learning_curve"] = learning_curve
        if feature_importance:
            reg_summary["diagnostics_charts/feature_importance"] = feature_importance
        if residuals:
            reg_summary["diagnostics_charts/residuals"] = residuals
        if prediction_error:
            reg_summary["diagnostics_charts/prediction_error"] = prediction_error
        if cooks_distance:
            reg_summary["diagnostics_charts/cooks_distance"] = cooks_distance

    return reg_summary


def create_classifier_summary(classifier, X_train, X_test, y_train, y_test, nrows=1000, log_charts=True):
    """Creates scikit-learn classifier summary.

    The summary includes:

    - all classifier parameters,
    - pickled estimator (model),
    - test predictions,
    - test predictions probabilities,
    - test scores,
    - model performance visualizations.

    The classifier should be fitted before calling this function.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.
        nrows (`int`, optional): Log first `nrows` rows of test predictions and prediction probabilities.
        log_charts (`bool`, optional): Whether to calculate and log chart visualizations.
            Note: Calculating visualizations is potentially expensive depending on input data and classifier, and
            may take some time to finish. This is equivalent to calling the following functions from this module:
            `create_classification_report_chart()`, `create_confusion_matrix_chart()`, `create_roc_auc_chart()`,
            `create_precision_recall_chart()`, and `create_class_prediction_error_chart()`.

    Returns:
        `dict` with all summary items.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["random_forest/summary"] = npt_utils.create_classifier_summary(rfc, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    cls_summary = dict()

    cls_summary["all_params"] = stringify_unsupported(get_estimator_params(classifier))
    cls_summary["pickled_model"] = get_pickled_model(classifier)

    y_pred = classifier.predict(X_test)

    cls_summary["test"] = {
        "preds": get_test_preds(classifier, X_test, y_test, y_pred=y_pred, nrows=nrows),
        "preds_proba": get_test_preds_proba(classifier, X_test, nrows=nrows),
        "scores": get_scores(classifier, X_test, y_test, y_pred=y_pred),
    }

    cls_summary["integration/about/neptune-sklearn"] = __version__

    if log_charts:
        classification_report = create_classification_report_chart(classifier, X_train, X_test, y_train, y_test)
        confusion_matrix = create_confusion_matrix_chart(classifier, X_train, X_test, y_train, y_test)
        roc_auc = create_roc_auc_chart(classifier, X_train, X_test, y_train, y_test)
        precision_recall = create_precision_recall_chart(classifier, X_test, y_test)
        class_prediction_error = create_class_prediction_error_chart(classifier, X_train, X_test, y_train, y_test)

        if classification_report:
            cls_summary["diagnostics_charts/classification_report"] = classification_report
        if confusion_matrix:
            cls_summary["diagnostics_charts/confusion_matrix"] = confusion_matrix
        if roc_auc:
            cls_summary["diagnostics_charts/ROC_AUC"] = roc_auc
        if precision_recall:
            cls_summary["diagnostics_charts/precision_recall"] = precision_recall
        if class_prediction_error:
            cls_summary["diagnostics_charts/class_prediction_error"] = class_prediction_error

    return cls_summary


def create_kmeans_summary(model, X, nrows=1000, **kwargs):
    """Creates scikit-learn k-means summary.

    Fits KMeans model to data and logs:

    - all KMeans parameters,
    - pickled estimator (model),
    - cluster labels,
    - clustering visualizations: KMeans elbow chart and silhouette coefficients chart.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        nrows (`int`, optional): Number of rows to log in the cluster labels.
        kwargs: KMeans parameters.

    Returns:
        `dict` with all summary items.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/summary"] = npt_utils.create_kmeans_summary(km, X)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "model should be sklearn KMeans instance"

    kmeans_summary = dict()
    model.set_params(**kwargs)

    kmeans_summary["all_params"] = stringify_unsupported(get_estimator_params(model))
    kmeans_summary["pickled_model"] = get_pickled_model(model)
    kmeans_summary["cluster_labels"] = get_cluster_labels(model, X, nrows=nrows, **kwargs)
    kmeans_summary["integration/about/neptune-sklearn"] = __version__

    kelbow = create_kelbow_chart(model, X, **kwargs)
    silhouette = create_silhouette_chart(model, X, **kwargs)

    if kelbow:
        kmeans_summary["diagnostics_charts/kelbow"] = kelbow
    if silhouette:
        kmeans_summary["diagnostics_charts/silhouette"] = silhouette

    return kmeans_summary


def get_estimator_params(estimator):
    """Returns estimator parameters.

    Args:
        estimator (`estimator`): Scikit-learn estimator from which to log parameters.

    Returns:
        `dict` with all parameters mapped to their values.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()

        run = neptune.init_run()
        run["estimator/params"] = npt_utils.get_estimator_params(rfr)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(estimator, BaseEstimator), "Estimator should be a sklearn estimator."

    return estimator.get_params()


def get_pickled_model(estimator):
    """Returns pickled estimator.

    Args:
        estimator (`estimator`): Scikit-learn estimator to pickle.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()

        run = neptune.init_run()
        run["estimator/pickled_model"] = npt_utils.get_pickled_model(rfr)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert (
        is_regressor(estimator) or is_classifier(estimator) or isinstance(estimator, KMeans)
    ), "Estimator should be sklearn regressor, classifier, or Kmeans instance."

    return File.as_pickle(estimator)


def get_test_preds(estimator, X_test, y_test, y_pred=None, nrows=1000):
    """Returns test predictions.

    The estimator should be fitted before calling this function.

    Args:
        estimator (`estimator`): Scikit-learn estimator to compute predictions.
        X_test (`ndarray`): Testing data matrix.
        y_test (`ndarray`): Target for testing.
        y_pred (`ndarray`, optional): Estimator predictions on test data.
            If you pass y_pred, then predictions are not computed from X_test data.
        nrows (`int`, optional): Number of rows to log.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()

        run = neptune.init_run()
        run["estimator/pickled_model"] = npt_utils.compute_test_preds(rfr, X_test, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(estimator) or is_classifier(estimator), "Estimator should be sklearn regressor or classifier."
    assert isinstance(nrows, int), "nrows should be integer, {} was passed".format(type(nrows))

    preds = None

    if y_pred is None:
        y_pred = estimator.predict(X_test)

    # single output
    if len(y_pred.shape) == 1:
        df = pd.DataFrame(data={"y_true": y_test, "y_pred": y_pred})
        df = df.head(n=nrows)
        preds = File.as_html(df)
    # multi output
    if len(y_pred.shape) == 2:
        df = pd.DataFrame()
        for j in range(y_pred.shape[1]):
            df["y_test_output_{}".format(j)] = y_test[:, j]
            df["y_pred_output_{}".format(j)] = y_pred[:, j]
        df = df.head(n=nrows)
        preds = File.as_html(df)

    return preds


def get_test_preds_proba(classifier, X_test=None, y_pred_proba=None, nrows=1000):
    """Returns probabilities of test predictions.

    The estimator should be fitted before calling this function.

    Args:
        classifier (`classifier`): Scikit-learn classifier to compute prediction probabilities.
        X_test (`ndarray`): Testing data matrix.
            If you pass this argument, then probabilities are computed from those data.
        y_pred_proba (`ndarray`, optional): Classifier predictions probabilities on test data.
            If you pass this argument, then probabilities are not computed from X_test data.
        nrows (`int`, optional): Number of rows to log.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["estimator/pickled_model"] = npt_utils.compute_test_preds(rfc, X_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "Classifier should be sklearn classifier."
    assert isinstance(nrows, int), "nrows should be integer, {} was passed".format(type(nrows))

    if X_test is not None and y_pred_proba is not None:
        raise ValueError("X_test and y_pred_proba are mutually exclusive")
    if X_test is None and y_pred_proba is None:
        raise ValueError("X_test or y_pred_proba is required")

    if y_pred_proba is None:
        try:
            y_pred_proba = classifier.predict_proba(X_test)
        except Exception as e:
            print("This classifier does not provide predictions probabilities. Error: {}".format(e))
            return

    df = pd.DataFrame(data=y_pred_proba, columns=classifier.classes_)
    df = df.head(n=nrows)

    return File.as_html(df)


def get_scores(estimator, X, y, y_pred=None):
    """Returns estimator scores on X.

    If you pass y_pred, then predictions are not computed from X and y data.

    The estimator should be fitted before calling this function.

    **Regressor**

    For regressors that output a single value, the following scores are logged:

    - explained variance
    - max error
    - mean absolute error
    - r2

    For multi-output regressor:

    - r2

    **Classifier**

    For a classifier, the following scores are logged:

    - precision
    - recall
    - f beta score
    - support

    Args:
        estimator (`estimator`): Scikit-learn estimator to compute scores.
        X (`ndarray`): Data matrix.
        y (`ndarray`): Target for testing.
        y_pred (`ndarray`, optional): Estimator predictions on data.

    Returns:
        `dict` with scores that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["estimator/scores"] = npt_utils.get_scores(rfc, X, y)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(estimator) or is_classifier(estimator), "Estimator should be sklearn regressor or classifier."

    scores_dict = {}

    if y_pred is None:
        y_pred = estimator.predict(X)

    if is_regressor(estimator):
        # single output
        if len(y_pred.shape) == 1:
            evs = explained_variance_score(y, y_pred)
            me = max_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            scores_dict["explained_variance_score"] = evs
            scores_dict["max_error"] = me
            scores_dict["mean_absolute_error"] = mae
            scores_dict["r2_score"] = r2

        # multi output
        if len(y_pred.shape) == 2:
            r2 = estimator.score(X, y)
            scores_dict["r2_score"] = r2

    elif is_classifier(estimator):
        precision, recall, fbeta_score, support = precision_recall_fscore_support(y, y_pred)
        for i, value in enumerate(precision):
            scores_dict["class_{}".format(i)] = {
                "precision": value,
                "recall": recall[i],
                "fbeta_score": fbeta_score[i],
                "support": support[i],
            }
    return scores_dict


def create_learning_curve_chart(regressor, X_train, y_train):
    """Creates learning curve chart.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        y_train (`ndarray`): The regression target for training.

    Returns:

        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.new.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/learning_curve"] = npt_utils.create_learning_curve_chart(rfr, X_train, y_train)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        plot_learning_curve(regressor, X_train, y_train, ax=ax)

        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log learning curve chart. Error: {}".format(e))

    return chart


def create_feature_importance_chart(regressor, X_train, y_train):
    """Creates feature importance chart.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        y_train (`ndarray`): The regression target for training.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/feature_importance"] = npt_utils.create_feature_importance_chart(rfr, X_train, y_train)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = FeatureImportances(regressor, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.finalize()

        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log feature importance chart. Error: {}".format(e))

    return chart


def create_residuals_chart(regressor, X_train, X_test, y_train, y_test):
    """Creates residuals chart.

    Args:
        regressor (`regressor`): Fitted scikit-learn regressor object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The regression target for training.
        y_test (`ndarray`): The regression target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.new.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/residuals"] = npt_utils.create_residuals_chart(rfr, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ResidualsPlot(regressor, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log residuals chart. Error: {}".format(e))

    return chart


def create_prediction_error_chart(regressor, X_train, X_test, y_train, y_test):
    """Creates prediction error chart.

    Args:
        regressor (`regressor`): Fitted sklearn regressor object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The regression target for training.
        y_test (`ndarray`): The regression target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.new.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["prediction_error"] = npt_utils.create_prediction_error_chart(rfr, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = PredictionError(regressor, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log prediction error chart. Error: {}".format(e))

    return chart


def create_cooks_distance_chart(regressor, X_train, y_train):
    """Creates cooks distance chart.

    Args:
        regressor (`regressor`): Fitted sklearn regressor object
        X_train (`ndarray`): Training data matrix
        y_train (`ndarray`): The regression target for training

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfr = RandomForestRegressor()
        rfr.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/cooks_distance"] = npt_utils.create_cooks_distance_chart(rfr, X_train, y_train)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_regressor(regressor), "regressor should be sklearn regressor."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = CooksDistance(ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log cooks distance chart. Error: {}".format(e))

    return chart


def create_classification_report_chart(classifier, X_train, X_test, y_train, y_test):
    """Creates classification report chart.

    Args:
        classifier (`classifier`): Fitted sklearn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/classification_report"] = npt_utils.create_classification_report_chart(
            rfc, X_train, X_test, y_train, y_test
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ClassificationReport(classifier, support=True, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log Classification Report chart. Error: {}".format(e))

    return chart


def create_confusion_matrix_chart(classifier, X_train, X_test, y_train, y_test):
    """Creates confusion matrix.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/confusion_matrix"] = npt_utils.create_confusion_matrix_chart(
            rfc, X_train, X_test, y_train, y_test
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ConfusionMatrix(classifier, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log Confusion Matrix chart. Error: {}".format(e))

    return chart


def create_roc_auc_chart(classifier, X_train, X_test, y_train, y_test):
    """Creates ROC-AUC chart.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/roc_auc"] = npt_utils.create_roc_auc_chart(rfc, X_train, X_test, y_train, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ROCAUC(classifier, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log ROC-AUC chart. Error {}".format(e))

    return chart


def create_precision_recall_chart(classifier, X_test, y_test, y_pred_proba=None):
    """Creates precision-recall chart.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_test (`ndarray`): Testing data matrix.
        y_test (`ndarray`): The classification target for testing.
        y_pred_proba (`ndarray`, optional): Classifier predictions probabilities on test data.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/precision_recall"] = npt_utils.create_precision_recall_chart(rfc, X_test, y_test)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    if y_pred_proba is None:
        try:
            y_pred_proba = classifier.predict_proba(X_test)
        except Exception as e:
            print(
                "Did not log Precision-Recall chart: this classifier does not provide predictions probabilities."
                "Error {}".format(e)
            )
            return chart

    try:
        fig, ax = plt.subplots()
        plot_precision_recall(y_test, y_pred_proba, ax=ax)
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log Precision-Recall chart. Error {}".format(e))

    return chart


def create_class_prediction_error_chart(classifier, X_train, X_test, y_train, y_test):
    """Creates class prediction error chart.

    Args:
        classifier (`classifier`): Fitted scikit-learn classifier object.
        X_train (`ndarray`): Training data matrix.
        X_test (`ndarray`): Testing data matrix.
        y_train (`ndarray`): The classification target for training.
        y_test (`ndarray`): The classification target for testing.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)

        run = neptune.init_run()
        run["visuals/class_prediction_error"] = npt_utils.create_class_prediction_error_chart(
            rfc, X_train, X_test, y_train, y_test
        )

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert is_classifier(classifier), "classifier should be sklearn classifier."

    chart = None

    try:
        fig, ax = plt.subplots()
        visualizer = ClassPredictionError(classifier, is_fitted=True, ax=ax)
        visualizer.fit(X_train, y_train)
        visualizer.score(X_test, y_test)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log Class Prediction Error chart. Error {}".format(e))

    return chart


def get_cluster_labels(model, X, nrows=1000, **kwargs):
    """Logs the index of the cluster label each sample belongs to.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        nrows (`int`, optional): Number of rows to log.
        kwargs: KMeans parameters.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/cluster_labels"] = npt_utils.get_cluster_labels(km, X)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "Model should be sklearn KMeans instance."
    assert isinstance(nrows, int), "nrows should be integer, {} was passed".format(type(nrows))

    model.set_params(**kwargs)
    labels = model.fit_predict(X)
    df = pd.DataFrame(data={"cluster_labels": labels})
    df = df.head(n=nrows)

    return File.as_html(df)


def create_kelbow_chart(model, X, **kwargs):
    """Creates K-elbow chart for KMeans clusterer.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        kwargs: KMeans parameters.

    Returns:
        `neptune.types.File` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/kelbow"] = npt_utils.create_kelbow_chart(km, X)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "Model should be sklearn KMeans instance."

    chart = None

    model.set_params(**kwargs)

    if "n_clusters" in kwargs:
        k = kwargs["n_clusters"]
    else:
        k = 10

    try:
        fig, ax = plt.subplots()
        visualizer = KElbowVisualizer(model, k=k, ax=ax)
        visualizer.fit(X)
        visualizer.finalize()
        chart = File.as_image(fig)
        plt.close(fig)
    except Exception as e:
        print("Did not log KMeans elbow chart. Error {}".format(e))

    return chart


def create_silhouette_chart(model, X, **kwargs):
    """Creates silhouette coefficients charts for KMeans clusterer.

    Charts are computed for j = 2, 3, ..., n_clusters.

    Args:
        model (`KMeans`): KMeans object.
        X (`ndarray`): Training instances to cluster.
        kwargs: KMeans parameters.

    Returns:
        `neptune.types.FileSeries` object that you can log to the run.

    Example:
        import neptune
        import neptune.integrations.sklearn as npt_utils

        km = KMeans(n_init=11, max_iter=270)
        X, y = make_blobs(n_samples=579, n_features=17, centers=7, random_state=28743)

        run = neptune.init_run()
        run["kmeans/silhouette"] = npt_utils.create_silhouette_chart(km, X, n_clusters=12)

    For more, see the docs:
        Tutorial: https://docs.neptune.ai/integrations/sklearn/
        API reference: https://docs.neptune.ai/api/integrations/sklearn/
    """
    assert isinstance(model, KMeans), "Model should be sklearn KMeans instance."

    charts = []

    model.set_params(**kwargs)

    n_clusters = model.get_params()["n_clusters"]

    for j in range(2, n_clusters + 1):
        model.set_params(**{"n_clusters": j})
        model.fit(X)

        try:
            fig, ax = plt.subplots()
            visualizer = SilhouetteVisualizer(model, is_fitted=True, ax=ax)
            visualizer.fit(X)
            visualizer.finalize()
            charts.append(File.as_image(fig))
            plt.close(fig)
        except Exception as e:
            print("Did not log Silhouette Coefficients chart. Error {}".format(e))

    return FileSeries(charts)
