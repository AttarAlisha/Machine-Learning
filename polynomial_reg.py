import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample data (x and y values)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])

# Degree of the polynomial (change this to experiment with higher-degree polynomials)
degree = 2

# Transform the input data to include polynomial terms up to the specified degree
poly_features = PolynomialFeatures(degree=degree)
x_poly = poly_features.fit_transform(x)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(x_poly, y)

# Predict y values based on the model
y_pred = model.predict(x_poly)

# Plotting the original data points and the polynomial regression curve
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label=f'{degree}-degree Polynomial fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print the coefficients and intercept
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
