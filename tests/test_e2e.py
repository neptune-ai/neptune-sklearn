try:
    from neptune import (
        Run,
        init_run,
    )
except ImportError:
    from neptune.new import Run, init_run

import matplotlib as mpl
import pytest
from numpy import array_equal
from sklearn.cluster import KMeans
from sklearn.dummy import (
    DummyClassifier,
    DummyRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import neptune_sklearn as npt_utils

mpl.use("agg")


def test_classifier_summary(iris):
    with init_run() as run:
        model = DummyClassifier()
        model.fit(iris.x_train, iris.y_train)

        run["summary"] = npt_utils.create_classifier_summary(
            model, iris.x_train, iris.x_test, iris.y_train, iris.y_test
        )

        run.wait()
        validate_run(run, log_charts=True)


def test_regressor_summary(diabetes):
    with init_run() as run:
        model = LinearRegression()
        model.fit(diabetes.x_train, diabetes.y_train)

        original_coef = model.coef_

        run["summary"] = npt_utils.create_regressor_summary(
            model, diabetes.x_train, diabetes.x_test, diabetes.y_train, diabetes.y_test
        )

        assert array_equal(model.coef_, original_coef), "Original model coefficients modified."

        run.wait()
        validate_run(run, log_charts=True)


def test_kmeans_summary(iris):
    with init_run() as run:
        model = KMeans()
        model.fit(iris.x)

        run["summary"] = npt_utils.create_kmeans_summary(model, iris.x, n_clusters=3)

        run.wait()
        validate_run(run, log_charts=True)


@pytest.mark.filterwarnings("error::neptune.common.warnings.NeptuneUnsupportedType")
def test_unsupported_object(diabetes):
    """This method checks if Neptune throws a `NeptuneUnsupportedType` warning if expected metadata
    is not found or skips trying to log such metadata"""

    with init_run() as run:
        model = DummyRegressor()
        model.fit(diabetes.x_train, diabetes.y_train)

        param_grid = {
            "strategy": ["mean", "median", "quantile"],
            "quantile": [0.1, 0.5, 1.0],
        }

        grid_cv = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", cv=2).fit(
            diabetes.x_train, diabetes.y_train
        )

        run["regressor_summary"] = npt_utils.create_regressor_summary(
            grid_cv,
            diabetes.x_train,
            diabetes.x_test,
            diabetes.y_train,
            diabetes.y_test,
        )

        run.wait()


def validate_run(run: Run, log_charts: bool) -> None:
    assert run.exists("summary/all_params")
    assert run.exists("summary/pickled_model")
    assert run.exists("summary/integration/about/neptune-sklearn")

    if log_charts:
        assert run.exists("summary/diagnostics_charts")
