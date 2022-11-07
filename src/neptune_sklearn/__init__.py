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
    "__version__",
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

import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import (
        PackageNotFoundError,
        version,
    )
else:
    from importlib_metadata import (
        PackageNotFoundError,
        version,
    )

from neptune_sklearn.impl import (
    create_class_prediction_error_chart,
    create_classification_report_chart,
    create_classifier_summary,
    create_confusion_matrix_chart,
    create_cooks_distance_chart,
    create_feature_importance_chart,
    create_kelbow_chart,
    create_kmeans_summary,
    create_learning_curve_chart,
    create_precision_recall_chart,
    create_prediction_error_chart,
    create_regressor_summary,
    create_residuals_chart,
    create_roc_auc_chart,
    create_silhouette_chart,
    get_cluster_labels,
    get_estimator_params,
    get_pickled_model,
    get_scores,
    get_test_preds,
    get_test_preds_proba,
)

try:
    __version__ = version("neptune-sklearn")
except PackageNotFoundError:
    # package is not installed
    pass
