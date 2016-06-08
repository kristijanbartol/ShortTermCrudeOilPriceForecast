import numpy as np
from ml_metrics import quadratic_weighted_kappa


class Evaluator:

    @staticmethod
    def rmse(yhat, y):
        for i in range(0, len(yhat)):
            print (str(yhat[i]) + ', ' + str(y[i]))
        return np.sqrt((yhat - y) ** 2).mean()
