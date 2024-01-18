from dataclasses import dataclass
from typing import Optional

import numpy as np
from pytest import fixture
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


_IRIS_DATASET: Optional[Bunch] = None
_DIABETES_DATASET: Optional[Bunch] = None


@fixture(scope="session")
def iris() -> Dataset:
    global _IRIS_DATASET
    if _IRIS_DATASET is None:
        _IRIS_DATASET = datasets.load_iris()
    x = _IRIS_DATASET.data[:, :2]
    y = _IRIS_DATASET.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    return Dataset(x, y, x_train, x_test, y_train, y_test)


@fixture(scope="session")
def diabetes() -> Dataset:
    global _DIABETES_DATASET
    if _DIABETES_DATASET is None:
        _DIABETES_DATASET = datasets.load_diabetes(return_X_y=True)
    x, y = _DIABETES_DATASET

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    return Dataset(x, y, x_train, x_test, y_train, y_test)
