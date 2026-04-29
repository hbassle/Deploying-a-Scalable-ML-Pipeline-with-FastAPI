import pytest
import numpy as np
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
from sklearn.linear_model import LogisticRegression

def test_compute_model_metrics():
    """
    Validates compute_model_metrics() returns 
    precision, recall, and f beta 
    as values between 0 and 1 inclusive

    Input: None
    Unit Test Values:
    y = np.array([0,1,1])
    preds = np.array([1,1,0])

    Return: AssertionError()
    """
    y = np.array([0,1,1])
    preds = np.array([1,1,0])

    p, r, f = compute_model_metrics(y,preds)

    assert 0 <= p <= 1, f"Precision value {p} is not between 0 and 1"
    assert 0 <= r <= 1, f"Recall value {r} is not between 0 and 1"
    assert 0 <= f <= 1, f"F Beta value {f} is not between 0 and 1"

def test_train_model():
    """
    Validates train_model() returns 
    LogisticRegression model that 
    has been fit

    Input: None
    Unit Test Values:
    X_train = np.array([
        [0,1],
        [1,0],
        [1,1],
        [0,0]
    ])
    y_train = np.array([0,0,1,1])

    Return: AssertionError()
    """
    X_train = np.array([
        [0,1],
        [1,0],
        [1,1],
        [0,0]
    ])
    y_train = np.array([0,0,1,1])

    model = train_model(X_train, y_train)

    mt = type(model).__name__
    assert mt=='LogisticRegression', f"Model type {mt} is not LogisticRegression"

    if model.n_features_in_:
        fit = True
    else:
        fit=False

    assert fit==True, "Model has not been fit"

def test_inference():
    """
    Validates that inference function returns 
    predictions for the fitted model.

    Input: None
    Unit Test Values:
    X_train = np.array([
        [0,1],
        [1,0],
        [1,1],
        [0,0]
    ])
    y_train = np.array([0,0,1,1])
    X_test = np.array([
        [1,1],
        [0,0],
        [0,1]
    ])

    Return: AssertionError()
 """
    X_train = np.array([
        [0,1],
        [1,0],
        [1,1],
        [0,0]
    ])
    y_train = np.array([0,0,1,1])

    # Not using train_model() to ensure 
    # a single function is unit tested
    model = LogisticRegression()
    model.fit(X_train,y_train)

    X_test = np.array([
        [1,1],
        [0,0],
        [0,1]
    ])
    preds = inference(model, X_test)
    assert preds.shape[0]==X_test.shape[0], f"Expected {X_test.shape[0]} values. Inference returned {preds.shape[0]}"
    assert preds.ndim == 1, f"Expected 1D array. Inference returned {preds.ndim}D array"
    assert np.isin(preds,[0,1]).all(), f"Output contains non-binary values. Expected 0 and 1 only."
