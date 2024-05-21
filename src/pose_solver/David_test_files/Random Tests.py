import cv2
import numpy as np

# Average rotation vector
avg_rotation_vector = np.array([np.pi/2, np.pi/2, 0])

# Convert to a rotation matrix
rotation_matrix, _ = cv2.Rodrigues(avg_rotation_vector)
print("Rotation Matrix from Averaged Vector:")
print(rotation_matrix)
