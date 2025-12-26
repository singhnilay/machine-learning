import numpy as np
from model import *

class Optimizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, model, djdw, djdb):

        model.w = model.w - self.alpha * djdw
        model.b = model.b - self.alpha * djdb

