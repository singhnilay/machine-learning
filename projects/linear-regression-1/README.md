# Linear Regression with Gradient Descent (From Scratch)

This project implements **univariate linear regression** using **gradient descent** from scratch in Python, without using machine learning libraries like `scikit-learn`.

The goal is to learn the parameters of a linear model:

\[
f(x) = wx + b
\]

that best fits a small training dataset.

---

## 📌 Dataset

We use a very small dataset with one feature:

```python
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
