"""Select Elastic Net settings using only the samples supplied for fitting."""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_elastic_net(
    features: np.ndarray, labels: np.ndarray, *, positive: bool, folds: int, seed: int
) -> GridSearchCV:
    values, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(values, [0, 1]) or counts.min() < folds:
        raise ValueError(f"Each of the two classes (0, 1) needs at least {folds} fitting subjects.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), ElasticNet(positive=positive, max_iter=20000, tol=1e-4))
    grid = GridSearchCV(
        model,
        {
            "elasticnet__alpha": [0.001, 0.01, 0.05, 0.1],
            "elasticnet__l1_ratio": [0.1, 0.5, 0.9],
        },
        scoring="neg_mean_squared_error",
        cv=list(cv.split(features, labels)),
        error_score="raise",
    )
    return grid.fit(features, labels)
