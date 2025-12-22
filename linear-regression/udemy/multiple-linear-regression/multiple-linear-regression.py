import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading data
df = pd.read_csv('/Users/ayenilay/Projects/machine-learning/linear-regression/udemy/multiple-linear-regression/50_Startups.csv')
X = df.iloc[:, : -1]
y = df.iloc[:,-1]
# preparing data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# splitting into test and train data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

print(y_test)
# fitting the linear regression to the training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_test_arr = np.array(y_test).reshape(-1, 1)
y_pred = lr.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(-1, 1), y_test_arr), axis=1))
