import numpy as np
import matplotlib.pyplot as plt

# Sample data (x and y values)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Number of data points
n = len(x)

# Calculate the slope (m) and intercept (b)
m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
b = (np.sum(y) - m * np.sum(x)) / n

# Predicted y values (y = mx + b)
y_pred = m * x + b

# Print the slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Plotting the data points and the regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
