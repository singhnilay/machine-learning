import numpy as np

class CostCalculator:
    def __init__(self):
        pass

    def compute_cost(self, y_predict, y):
        m = len(y)

        cost = (y_predict - y)** 2

        return np.sum(cost) / (2 * m)

