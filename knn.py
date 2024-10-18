import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (features and labels)
# Let's use a simple dataset with two features
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [5.0, 5.0], [6.0, 7.0],
              [1.5, 2.5], [3.5, 4.5], [4.0, 5.0], [5.5, 6.0], [6.5, 7.5]])
y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the K-Nearest Neighbors model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print predictions
print(f"Predicted labels: {y_pred}")
print(f"Actual labels: {y_test}")
