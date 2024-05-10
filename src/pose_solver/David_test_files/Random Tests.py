import cv2
import numpy as np

# Create a blank image
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Camera intrinsic parameters (e.g., from calibration)
# Ensuring that camera_matrix is a 3x3 floating-point matrix
camera_matrix = np.array([[320, 0, 200],
                          [0, 320, 200],
                          [0, 0, 1]], dtype=np.float32)

# Assume no lens distortion
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Object's rotation and translation vectors
# Ensuring rvec and tvec are 3x1 floating-point vectors
rvec = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1)
tvec = np.array([0, 0, 1], dtype=np.float32).reshape(3, 1)

# Length of the axes to be drawn
length = 100

# Draw the axes
cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length)

# Display the image
cv2.imshow('Frame Axes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
