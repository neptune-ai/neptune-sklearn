try:
    from neptune import init_run
except ImportError:
    from neptune.new import init_run

import pytest
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)

import neptune_sklearn as npt_utils


def test_classifier_summary():
    run = init_run()

    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = LogisticRegression(C=1e5)
    model.fit(X_train, y_train)

    run["summary"] = npt_utils.create_classifier_summary(model, X_train, X_test, y_train, y_test)

    run.wait()
    validate_run(run, log_charts=True)


def test_regressor_summary():
    run = init_run()

    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = LinearRegression()
    model.fit(X_train, y_train)

    run["summary"] = npt_utils.create_regressor_summary(model, X_train, X_test, y_train, y_test)

    run.wait()
    validate_run(run, log_charts=True)


def test_kmeans_summary():
    run = init_run()

    iris = datasets.load_iris()
    X = iris.data[:, :2]

    model = KMeans()
    model.fit(X)

    run["summary"] = npt_utils.create_kmeans_summary(model, X, n_clusters=3)

    run.wait()
    validate_run(run, log_charts=True)


@pytest.mark.filterwarnings("error::neptune.common.warnings.NeptuneUnsupportedType")
def test_unsupported_object():
    """This method checks if Neptune throws a `NeptuneUnsupportedType` warning if expected metadata
    is not found or skips trying to log such metadata"""

    with init_run() as run:

        X, y = datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        model = LinearRegression()

        param_grid = {
            "copy_X": [True, False],
            "fit_intercept": [True, False],
        }

        X, y = datasets.fetch_california_housing(return_X_y=True)[:10]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        grid_cv = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", cv=2).fit(X_train, y_train)

        run["regressor_summary"] = npt_utils.create_regressor_summary(grid_cv, X_train, X_test, y_train, y_test)

        run.wait()


def validate_run(run, log_charts):
    assert run.exists("summary/all_params")
    assert run.exists("summary/pickled_model")
    assert run.exists("summary/integration/about/neptune-sklearn")

    if log_charts:
        assert run.exists("summary/diagnostics_charts")
