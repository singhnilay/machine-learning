import numpy as np

class GradientCalculator:
    def __init__(self):
        pass

    def compute_gradient(self, X, y_predict, y):
        m = len(y_predict)

        djdw = np.dot(X.T, (y_predict-y))/ m

        djdb = np.sum(y_predict - y) / m


        return djdw, djdb