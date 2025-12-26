
class Trainer:
    def __init__(self, model, cost_fn, gradient_fn, optimizer):
        self.model = model
        self.cost_fn = cost_fn
        self.gradient_fn = gradient_fn
        self.optimizer = optimizer

    def train(self, X, y, epochs):

        for epoch in range(epochs):
            y_pred = self.model.predict(X)

            loss = self.cost_fn.compute_cost(y_pred , y)

            djdw, djdb = self.gradient_fn.compute_gradient(X, y_pred, y)

            self.optimizer.step(self.model, djdw, djdb)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
