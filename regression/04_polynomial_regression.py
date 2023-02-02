import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('./datasets/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

print("X: \n")
print(X, "\n")
print("y: \n")
print(y, "\n")

# Fitting the Regression Model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

print("X_poly: \n")
print(X_poly, "\n")

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Linear Regression results
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Predicting a new result with Linear Regression
print("Predicting a new result with Linear Regression: \n")
print(lin_reg.predict([[6.5]]), "\n")

# Predicting a new result with Polynomial Regression
print("Predicting a new result with Polynomial Regression: \n")
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])), "\n")