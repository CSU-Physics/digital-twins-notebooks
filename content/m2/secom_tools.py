"""Helpers for part A of the Module 2 lab (Digital Twins for Semiconductor Manufacturing): the SECOM data.

SECOM: McCann, M. and Johnston, A. (2008), UCI Machine Learning Repository, https://doi.org/10.24432/C54305, CC BY 4.0.
1,567 production records with 590 anonymized signals each, and a pass or fail result of the in-house line test.

load_secom()     reads the two data files next to this module (data/secom.data.gz and data/secom_labels.data).
time_split()     splits the records by time: the earliest 70% for training, the latest 30% for testing.
evaluate()       fits a simple classifier on chosen signals of the training records and scores it on the test records.
screening_pipeline()  the same steps as evaluate(), with the screening filter and the ranking inside a scikit-learn pipeline.
"""
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "1.1.1"

DATA = Path(__file__).resolve().parent / "data"
TRAIN_FRACTION = 0.7


def load_secom(folder=DATA):
    """Read SECOM. Returns X, y, t, all with the same row index 0 to 1566, in time order:
      X  DataFrame of the 590 signals, columns s000 to s589 (the column number in the file), NaN where missing;
      y  Series named "fail": 1 for a record that failed the line test, 0 for a pass;
      t  Series named "time": the date and time of the line test.
    """
    folder = Path(folder)
    X = pd.read_csv(folder / "secom.data.gz", sep=" ", header=None, compression="gzip")
    X.columns = [f"s{j:03d}" for j in range(X.shape[1])]
    lab = pd.read_csv(folder / "secom_labels.data", sep=" ", header=None, names=["label", "time"])
    y = (lab["label"] == 1).astype(int).rename("fail")
    t = pd.to_datetime(lab["time"], format="%d/%m/%Y %H:%M:%S").rename("time")
    return X, y, t


def time_split(X, y, t, train_fraction=TRAIN_FRACTION):
    """Split by time: the earliest records train, the latest test. Returns X_train, X_test, y_train, y_test."""
    if not t.is_monotonic_increasing:
        raise ValueError("the records are not in time order")
    n_train = int(round(train_fraction * len(y)))
    return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


def _classifier():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Every step is fitted on the training records only, inside the pipeline: the median used to fill a missing
    # value, the mean and standard deviation used for scaling, and the classifier's weights.
    return [SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000)], make_pipeline


def _scores(model, X_test, y_test):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    y_test = np.asarray(y_test)
    return {"auc": float(roc_auc_score(y_test, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "accuracy": float(np.mean(pred == y_test)),
            "fails_found": int(np.sum((pred == 1) & (y_test == 1))), "fails": int(np.sum(y_test == 1))}


def evaluate(X_train, y_train, X_test, y_test, columns):
    """Fit fill-in, scaling and a balanced logistic regression on the chosen columns of the training records,
    then score the test records. Returns a dict: auc (area under the ROC curve: 0.5 is chance, 1 is perfect),
    balanced_accuracy (the mean of the share of passes and the share of fails called correctly), accuracy,
    fails_found and fails (how many test fails the classifier flagged, out of how many)."""
    steps, make_pipeline = _classifier()
    columns = list(columns)
    model = make_pipeline(*steps).fit(X_train[columns], np.asarray(y_train))
    return _scores(model, X_test[columns], y_test)


def screening_pipeline(keep_function, score_function=None, k=20):
    """A scikit-learn pipeline with the same steps, in the same order, as Steps 5 to 7 of the lab:
      1. keep_function(X) chooses the columns to keep (the missing-value and near-constant filter);
      2. score_function(X, y) ranks the kept columns, on the values as recorded (missing values skipped), and the k best stay;
      3. missing values are filled in with the median, the signals are scaled, and the balanced logistic regression is fitted.
    Every step learns from the rows the pipeline is fitted on, and only those, so cross-validation repeats the filter
    and the ranking inside every fold. keep_function returns a list of column names; score_function returns one score per column.

    Called with one function, screening_pipeline(score_function, k=20), as in notebooks older than version 2026-10-25, it
    returns the pipeline of secom_tools 1.0.0 (fill in, scale, keep the k best by the score, classify), so a saved older
    notebook still runs and gives the result it gave before."""
    from sklearn.base import BaseEstimator, TransformerMixin
    steps, make_pipeline = _classifier()
    if score_function is None:
        from sklearn.feature_selection import SelectKBest
        old_score_function = keep_function

        def score(A, b):
            A = pd.DataFrame(A)
            return np.asarray(old_score_function(A, pd.Series(np.asarray(b), index=A.index)), dtype=float)

        return make_pipeline(steps[0], steps[1], SelectKBest(score, k=k), steps[2])

    class KeepColumns(BaseEstimator, TransformerMixin):
        def __init__(self, keep_function=None):
            self.keep_function = keep_function

        def fit(self, X, y=None):
            self.columns_ = list(self.keep_function(pd.DataFrame(X)))
            return self

        def transform(self, X):
            return pd.DataFrame(X)[self.columns_]

    class TopScores(BaseEstimator, TransformerMixin):
        def __init__(self, score_function=None, k=20):
            self.score_function = score_function
            self.k = k

        def fit(self, X, y):
            X = pd.DataFrame(X)
            s = pd.Series(np.asarray(self.score_function(X, pd.Series(np.asarray(y), index=X.index)), dtype=float), index=X.columns)
            self.columns_ = list(s.sort_values(ascending=False).index[:self.k])
            return self

        def transform(self, X):
            return pd.DataFrame(X)[self.columns_]

    return make_pipeline(KeepColumns(keep_function), TopScores(score_function, k), *steps)
