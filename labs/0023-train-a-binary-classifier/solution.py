import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.

    Args:
        X_train: numpy array of shape (n_samples, 30)
        y_train: numpy array of shape (n_samples,)
        X_val: numpy array of shape (n_val, 30)
        y_val: numpy array of shape (n_val,)

    Returns:
        predict: callable that takes X (n, 30)
                 and returns y_pred (n,) of 0s and 1s
    """

    model = LogisticRegression(
        max_iter=1000
    )

    parameters = {
        "C": [0.1, 0.2, 0.01, 1, 2]
    }

    grid = GridSearchCV(
        model,
        param_grid=parameters,
        cv=3,
        scoring="f1"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    val_prediction = best_model.predict(X_val)

    def predict(X):
        return best_model.predict(X)

    return predict