import numpy as np

class cost:
    def __init__(self, x, y, w, b):
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = w
        self.b = b


    def compute_cost(self):
        m = len(self.y)
        total_cost = 0

        for i in range(m):
            f = self.w * self.x[i] + self.b
            cost_i = (f - self.y[i]) ** 2
            total_cost += cost_i

        return 1 / (2*m) * cost

