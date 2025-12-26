import numpy as np

from model import LinearRegressionModel
from cost import CostCalculator
from gradient import GradientCalculator
from optimizer import Optimizer
from trainer import Trainer

X = np.array([[1],
              [2],
              [3],
              [4],
              [5]])

y = np.array([3, 5, 7, 9, 11])

model = LinearRegressionModel(n_features=X.shape[1])
cost_fn = CostCalculator()
gradient_fn = GradientCalculator()
optimizer = Optimizer(alpha=0.1)

trainer = Trainer(
    model=model,
    cost_fn=cost_fn,
    gradient_fn=gradient_fn,
    optimizer=optimizer
)


trainer.train(X, y, epochs=1000)

X_test = np.array([[6], [7], [8]])
predictions = model.predict(X_test)

print("\nFinal parameters:")
print("Weights:", model.w)
print("Bias:", model.b)

print("\nPredictions:")
for x, y_hat in zip(X_test, predictions):
    print(f"x={x[0]} -> ŷ={y_hat:.2f}")
