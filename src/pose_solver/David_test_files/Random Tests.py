import numpy as np

# Sample RVEC vector values
rvec_values = np.array([0.2, 0.5, 0.8, 0.3])

# Create the relationship matrix
n = len(rvec_values)
relationship_matrix = np.zeros((n, n))

"""
# Fill in the relationship matrix
for i in range(n):
    for j in range(n):
        # Calculate the relationship value between the first element of the vectors
        relationship_matrix[i, j] = rvec_values[i] - rvec_values[j]
"""


print("Relationship Matrix:")
print(relationship_matrix)
