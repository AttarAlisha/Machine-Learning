import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Sample data (x and y values)
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

# Feature scaling (SVR is sensitive to the scale of input features)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Fit SVR model with the RBF kernel
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(x_scaled, y_scaled)

# Predict y values
y_pred_scaled = svr_model.predict(x_scaled)

# Inverse transform the predictions to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Plotting the original data points and the SVR regression curve
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, y_pred, color='red', label='SVR Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Print model details
print(f"Support Vectors: {svr_model.support_}")
print(f"Predicted Values: {y_pred.flatten()}")
