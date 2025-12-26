import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
df = pd.read_csv('/Users/ayenilay/Projects/machine-learning/regression/udemy/support-vector-regression/Position_Salaries.csv')
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(X), 1)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

#importing SVR model
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X,y)

#Predicting a new result
print(scaler_y.inverse_transform(svr.predict(scaler_x.transform([[6.5]])).reshape(-1,1)))

# Visualising Prediction Results

plt.scatter(scaler_x.inverse_transform(X), scaler_y.inverse_transform(y), color = 'red')
plt.plot(scaler_x.inverse_transform(X), scaler_y.inverse_transform(svr.predict(X).reshape(-1,1)))
plt.show()