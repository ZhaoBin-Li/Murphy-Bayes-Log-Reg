import numpy as np
import pandas as pd

import multinomial_bayes_logistic as mbl

def test_parkinsons():
    # Download and process data
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    y = df.values[:, 17]
    y = y.astype(int)
    X = np.delete(df.values, 17, 1)
    X = np.delete(X, 0, 1)
    n_samples, n_features = X.shape

    # Add bias column to the feature matrix
    B = np.ones((n_samples, n_features + 1))
    B[:, 1:] = X
    X = B

    # Perform feature scaling using mean normalization
    for col in range(1, n_features):
        v = X[:, col]
        mean = v.mean()
        std = v.std()
        X[:, col] = (X[:, col] - mean) / std

    # The data is divided into training and test sets
    TRAINING_PERCENTAGE = 0.7

    training_cnt = int(n_samples * TRAINING_PERCENTAGE)
    training_X = X[:training_cnt, :]
    training_y = y[:training_cnt]

    test_X = X[training_cnt:, :]
    test_y = y[training_cnt:]

    # Train the model
    n_classes = 2

    w_prior = np.zeros((n_classes, training_X.shape[1]))
    H_prior = np.eye((n_classes * (n_features + 1)), (n_classes * (n_features + 1)))

    w_posterior, H_posterior = mbl.fit(training_y,
                                       training_X,
                                       w_prior, H_prior)

    test_probs, preds, max_probs = mbl.get_bayes_point_probs(test_X, w_posterior)
    bayes_acc =  np.mean(preds == test_y)

    assert bayes_acc > 0.5

    test_probs, preds, max_probs = mbl.get_monte_carlo_probs(test_X, w_posterior, H_posterior)
    mc_acc = np.mean(preds == test_y)

    assert mc_acc > 0.5
    np.testing.assert_allclose(bayes_acc, mc_acc)
