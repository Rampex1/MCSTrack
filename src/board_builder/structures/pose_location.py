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

    def get_matrix(self):
        return self.__TMatrix


