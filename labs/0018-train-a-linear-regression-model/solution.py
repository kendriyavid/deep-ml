import numpy as np


def train(X, y, W, b):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    W = np.asarray(W, dtype=float)

    alphavals = [0.1, 0.2, 0.01, 0.001]
    epochs = 30

    m = y.shape[0]

    best_loss = np.inf
    best_W = None
    best_b = None

    for alpha in alphavals:

        np_W = W.copy()
        np_b = float(b)

        for epoch in range(epochs):

            y_pred = X @ np_W + np_b

            error = y_pred - y

            loss = np.mean(error ** 2)

            gradw = (X.T @ error) / m
            gradb = np.mean(error)

            np_W = np_W - alpha * gradw
            np_b = np_b - alpha * gradb

        if loss < best_loss:
            best_loss = loss
            best_W = np_W.copy()
            best_b = np_b

    return best_W, best_b