import numpy as np

# Example: Creating a 3x3 matrix with some data
n = 3
original_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create a new (n+1)x(n+1) matrix initialized to zeros
expanded_matrix = np.zeros((n+1, n+1), dtype=original_matrix.dtype)

# Copy the original matrix data into the new matrix
expanded_matrix[:n, :n] = original_matrix

# Output the result
print("Original Matrix:")
print(original_matrix)
print("\nExpanded Matrix:")
print(expanded_matrix)
