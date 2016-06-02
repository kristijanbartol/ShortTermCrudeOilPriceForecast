import numpy as np
from ml_metrics import quadratic_weighted_kappa


class Evaluator:

    @staticmethod
    def weighted_kappa(yhat, y):
        y = np.array(y)
        y = y.astype(int)
        yhat = np.array(yhat)
        yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
        return quadratic_weighted_kappa(yhat, y)
