import numpy as np

# Define the transformation matrix T for Detector A
TA = np.array([
    [0, -1 / np.sqrt(2), 1 / np.sqrt(2), -500],
    [-1, 0, 0, 0],
    [0, -1 / np.sqrt(2), -1 / np.sqrt(2), 500],
    [0, 0, 0, 1]
])

