import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('udemy/polynomial-linear-regression/Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

#plotting the scatter graph

plt.scatter(X, y, color = 'red')
plt.plot(X, lr.predict(X))
plt.show()

# applying polynomial regression 

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree = 6)

X_poly = pr.fit_transform(X)

lr2 = LinearRegression()

lr2.fit(X_poly, y)

# visualising polynomial regression

plt.scatter(X,y, color = 'red')
plt.plot(X, lr2.predict(X_poly), linestyle = '--')
plt.show()

# predicting results with polynomial regression

print(lr2.predict(pr.fit_transform([[6.5]])))