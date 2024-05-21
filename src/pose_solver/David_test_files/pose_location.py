import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseLocation:
    def __init__(self):
        self.__TMatrix = np.eye(4)

        self.__RMAT_list = []  # Rotation matrix
        self.__TVEC_list = []  # Translation vector

    def add_matrix(self, transformation_matrix):

        self.__RMAT_list.append(transformation_matrix[:3, :3])
        self.__TVEC_list.append(transformation_matrix[:3, 3])

        avg_translation = np.mean(self.__TVEC_list, axis=0)

        quaternions = [R.from_matrix(rot).as_quat() for rot in self.__RMAT_list]
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize the average quaternion
        avg_rotation = R.from_quat(avg_quat).as_matrix()

        self.__TMatrix[:3, :3] = avg_rotation
        self.__TMatrix[:3, 3] = avg_translation

    def get_TMatrix(self):
        return self.__TMatrix


"""
pose_location = PoseLocation()

# Define some transformation matrices
transform1 = np.eye(4)
transform1[:3, :3] = R.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix()
transform1[:3, 3] = [1, 2, 3]

transform2 = np.eye(4)
transform2[:3, :3] = R.from_euler('xyz', [0, 45, 0], degrees=True).as_matrix()
transform2[:3, 3] = [5, 2, 6]

transform3 = np.eye(4)
transform3[:3, :3] = R.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix()
transform3[:3, 3] = [6, 11, 9]

# Add these matrices to the PoseLocation
pose_location.add_matrix(transform1)
pose_location.add_matrix(transform2)
pose_location.add_matrix(transform3)

# Retrieve the averaged transformation matrix
avg_matrix = pose_location.get_TMatrix()

# Print the average transformation matrix
print("Averaged Transformation Matrix:")
print(avg_matrix)
"""