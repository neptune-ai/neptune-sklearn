## neptune-sklearn 2.1.2

### Changes
- Constraining scipy to `<1.12` ([#25](https://github.com/neptune-ai/neptune-sklearn/pull/25))

## neptune-sklearn 2.1.1

### Fixes
- `create_*_summary()` now does not throw a `NeptuneUnsupportedType` error if expected metadata is not found ([#21](https://github.com/neptune-ai/neptune-sklearn/pull/21))
- Fixed method names in docstrings ([#18](https://github.com/neptune-ai/neptune-sklearn/pull/18))

## neptune-sklearn 2.1.0

### Changes
- removed `neptune` and `neptune-client` from base requirements
- updated integration for compatibility with `neptune 1.X`

## neptune-sklearn 2.0.0

### Changes
- `create_kmeans_summary` is now saving the pickled model to Neptune.
- We use the `stringify_unsupported` to wrap the saved model parameters.

## neptune-sklearn 0.1.3

### Changes
- Moved neptune_sklearn package to src dir ([#5](https://github.com/neptune-ai/neptune-sklearn/pull/5))
- Poetry as a package builder ([#10](https://github.com/neptune-ai/neptune-sklearn/pull/10))

### Fixes
- Fixed imports from `neptune_sklearn.impl` -> now possible to import from `neptune_sklearn` ([#7](https://github.com/neptune-ai/neptune-sklearn/pull/7))

## neptune-sklearn 0.1.2

### Changes
- `get_estimator_params` works now with any scikit-learn estimator ([#3](https://github.com/neptune-ai/neptune-sklearn/pull/3))
