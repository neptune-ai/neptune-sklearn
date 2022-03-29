# Neptune + Scikit-Learn Integration

Experiment tracking, model registry, data versioning, and live model monitoring for Scikit-Learn (Sklearn) trained models.

## What will you get with this integration? 

* Log, display, organize, and compare ML experiments in a single place
* Version, store, manage, and query trained models, and model building metadata
* Record and monitor model training, evaluation, or production runs live

## What will be logged to Neptune?

* classifier and regressor parameters,
* pickled model,
* test predictions,
* test predictions probabilities,
* test scores,
* classifier and regressor visualizations, like confusion matrix, precision-recall chart, and feature importance chart,
* KMeans cluster labels and clustering visualizations,
* metadata including git summary info.
* [other metadata](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)

![image](https://user-images.githubusercontent.com/97611089/160642485-afca99da-9f7b-4d80-b0be-810c9d5770e5.png)
*Confusion matrix logged to Neptune*


## Resources

* [Documentation](https://docs.neptune.ai/integrations-and-supported-tools/model-training/sklearn)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/sklearn/scripts/Neptune_Scikit_learn_classification.py)
* [Runs logged in the Neptune app](https://app.neptune.ai/o/common/org/sklearn-integration/e/SKLEAR-95/all)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/sklearn/notebooks/Neptune_Scikit_learn.ipynb)

## Minimal example

```python
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils

run = neptune.init(project='common/sklearn-integration',
                   api_token='ANONYMOUS',
                   name='regression-example',
                   tags=['RandomForestRegressor', 'regression'])

parameters = {'n_estimators': 70,
              'max_depth': 7,
              'min_samples_split': 3}


rfr = RandomForestRegressor(**parameters)

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=28743)

rfr.fit(X_train, y_train)

run['rfr_summary'] = npt_utils.create_regressor_summary(rfr, X_train, X_test, y_train, y_test)

```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting-started/getting-help#frequently-asked-questions)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
