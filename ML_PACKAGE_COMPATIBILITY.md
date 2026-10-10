# scikit-learn / xgboost Compatibility Guide

This project has been adjusted to run with both an older and a newer machine-learning package combination.

## Verified package combinations

Old combination:

- `scikit-learn==1.0.2`
- `xgboost==1.6.2`

New combination:

- `scikit-learn==1.8.0`
- `xgboost==3.2.0`

## What was changed for compatibility

The main compatibility risk was `xgboost`.

Older code often used:

```python
XGBClassifier(use_label_encoder=False, ...)
```

This is tolerated by older `xgboost`, but newer `xgboost 3.x` reports that `use_label_encoder` is no longer used.

So the training script was kept version-safe by:

- removing `use_label_encoder=False`
- keeping stable parameters such as `objective="binary:logistic"` and `eval_metric="logloss"`
- preserving the `fix_xgboost_model(...)` compatibility patch when loading pickled models

The `scikit-learn` APIs used here are stable across these versions:

- `Pipeline`
- `StandardScaler`
- `SVC`
- `train_test_split`
- `accuracy_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `roc_auc_score`

## Which file is compatible

Main file:

- `stage2_pipeline.py`

This file has been tested with both package combinations above.

## How to check your current package versions

```cmd
python -c "import sklearn, xgboost; print('sklearn', sklearn.__version__); print('xgboost', xgboost.__version__)"
```

## How to upgrade from old packages to new packages

Use the Python interpreter you actually want to upgrade, then run:

```cmd
python -m pip install --upgrade scikit-learn xgboost
```

If you want to install into a specific interpreter, use:

```cmd
FULL_PATH_TO_PYTHON.exe -m pip install --upgrade scikit-learn xgboost
```

Example:

```cmd
C:\Users\33909\AppData\Local\Programs\Python\Python314\python.exe -m pip install --upgrade scikit-learn xgboost
```

## Important note

Upgrading only the packages is not always enough.

Newer package versions may require a newer Python version.

For example:

- `scikit-learn 1.8.0` does not work with old `Python 3.7`
- `xgboost 3.2.0` was installed successfully in the newer Python environment

So the real rule is:

- old Python can only use older compatible package versions
- new Python can use the newest package versions

## Recommended workflow

If your goal is just to run the project reliably:

1. Use the older compatible pair on the old environment.
2. Use the newest pair on the new environment.
3. Keep the same `stage2_pipeline.py` code for both.

## Tested status

The current script was run successfully with:

- old pair: `scikit-learn 1.0.2` + `xgboost 1.6.2`
- new pair: `scikit-learn 1.8.0` + `xgboost 3.2.0`

## Rollback example

If you need to go back to the older package pair:

```cmd
python -m pip install scikit-learn==1.0.2 xgboost==1.6.2
```
