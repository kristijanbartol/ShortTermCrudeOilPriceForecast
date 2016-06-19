import numpy as np

import sklearn

class Evaluator:

    @staticmethod
    def rmse(yhat, y):
        # for i in range(0, len(yhat)):
        #     print(str(yhat[i]) + ', ' + str(y[i]))
        return np.sqrt((yhat - y) ** 2).mean()

    @staticmethod
    def actual_diff(actual):
        return

    def rse(self, yhat, y):
        pass

    def coef_of_det(self, yhat, y):
        return 1 - self.rse(yhat, y)

