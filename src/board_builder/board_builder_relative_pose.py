import cv2
from cv2 import aruco
import numpy as np
import pose_location

from src.board_builder.pose_location import PoseLocation

class BoardBuilder:

    def __init__(self, relative_pose_matrix, marker_id_to_uuid, index_to_marker_id):
        self.relative_pose_matrix = relative_pose_matrix
        self.marker_id_to_uuid = marker_id_to_uuid
        self.index_to_marker_id = index_to_marker_id

        self._local_relative_pose_matrix = [[None for _ in range(len(self.relative_pose_matrix))] for _ in range(len(self.relative_pose_matrix))]

    def _find_matrix_input_index(self, pose_uuid, other_pose_uuid, marker_id_to_uuid, index_to_marker_id):
        """ Given two pose uuids, return their index location in the relative pose matrix """
        pose_id = -1
        other_pose_id = -1
        pose_index = -1
        other_pose_index = -1

        for id in marker_id_to_uuid:
            if marker_id_to_uuid[id] == pose_uuid:
                pose_id = id
            if marker_id_to_uuid[id] == other_pose_uuid:
                other_pose_id = id

        if pose_id != -1 and other_pose_id != -1:
            for index in index_to_marker_id:
                if index_to_marker_id[index] == pose_id:
                    pose_index = index
                if index_to_marker_id[index] == other_pose_id:
                    other_pose_index = index
            return pose_index, other_pose_index

        return None

    def _estimate_reference_to_not_visible(self, T_AB, T_BC):
        T_AC = np.dot(T_AB, T_BC)
        return T_AC


    def _find_new_reference(self):
        new_reference = PoseLocation()

        for i in range(len(self.relative_pose_matrix)):
            for j in range(len(self.relative_pose_matrix)):
                if self.relative_pose_matrix[i][j] is not None:
                    new_reference.add_matrix(self.relative_pose_matrix[i][j].get_TMatrix())

        return new_reference.get_TMatrix()

    def _calculate_relative_transform(self, T1, T2):
        """ Given transform T1 from reference to marker 1, and transfrom T2 from reference to marker 2, calculate the
        transform from T1 to T2"""
        T1_inv = np.linalg.inv(T1)
        relative_T = np.dot(T1_inv, T2)
        return relative_T

    def build_board(self):
        new_reference = self._find_new_reference()

        for i in range(len(self.relative_pose_matrix)):
            for j in range(len(self.relative_pose_matrix)):
                if self.relative_pose_matrix[i][j] is not None:
                    new_pose_location = PoseLocation()
                    new_matrix_from_reference = self._calculate_relative_transform(new_reference, self.relative_pose_matrix[i][j].get_TMatrix())
                    new_pose_location.add_matrix(new_matrix_from_reference)
                    self._local_relative_pose_matrix[i][j] = new_pose_location

        return self._local_relative_pose_matrix
