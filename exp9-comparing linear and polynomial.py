import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable
y = np.array([2, 4, 5, 4, 6])  # Dependent variable
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X, y)
poly_pred = poly_model.predict(X)
plt.scatter(X, y, label='Actual Data')
plt.plot(X, linear_pred, color='blue', label='Linear Regression')
plt.plot(X, poly_pred, color='red', label=f'Polynomial Regression (degree {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear vs Polynomial Regression')
plt.show()
