import numpy as np

class LinearRegressionModel:
    def __init__(self, n_features):
        self.n = n_features
        self.w = np.zeros(self.n)
        self.b = 0
    
    def predict(self, X):
        
        y = np.dot(X, self.w) + self.b
        return y