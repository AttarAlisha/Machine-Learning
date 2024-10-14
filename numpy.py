import numpy as np

# 1. Creating NumPy arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])
print("Array 1:", arr1)
print("Array 2:", arr2)

# 2. Element-wise addition
add_result = np.add(arr1, arr2)
print("\nElement-wise addition:", add_result)

# 3. Element-wise subtraction
sub_result = np.subtract(arr1, arr2)
print("Element-wise subtraction:", sub_result)

# 4. Element-wise multiplication
mul_result = np.multiply(arr1, arr2)
print("Element-wise multiplication:", mul_result)

# 5. Element-wise division
div_result = np.divide(arr1, arr2)
print("Element-wise division:", div_result)

# 6. Dot product of two arrays
dot_product = np.dot(arr1, arr2)
print("\nDot product:", dot_product)

# 7. Creating a 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nMatrix:\n", matrix)

# 8. Transpose of a matrix
matrix_transpose = np.transpose(matrix)
print("\nTranspose of matrix:\n", matrix_transpose)

# 9. Matrix multiplication
matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
matrix_multiplication = np.matmul(matrix, matrix2)
print("\nMatrix multiplication:\n", matrix_multiplication)

# 10. Mean and Standard Deviation of an array
mean = np.mean(arr1)
std_dev = np.std(arr1)
print("\nMean of arr1:", mean)
print("Standard deviation of arr1:", std_dev)

# 11. Reshaping an array
reshaped = np.reshape(arr1, (5, 1))  # Reshape 1D array to 2D (5 rows, 1 column)
print("\nReshaped array:\n", reshaped)
