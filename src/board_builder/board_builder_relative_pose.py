import cv2
from cv2 import aruco
import numpy as np

from src.board_builder.pose_location import PoseLocation

class BoardBuilder:

    def __init__(self, target_poses, relative_pose_matrix, marker_id_to_uuid, index_to_marker_id):
        self.target_poses = target_poses
        self.relative_pose_matrix = relative_pose_matrix
        self.marker_id_to_uuid = marker_id_to_uuid
        self. index_to_marker_id = index_to_marker_id

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


    def _calculate_corners_location(self, T_matrix, local_corners):
        corners_reference = np.zeros((4, 4))
        for i in range(4):
            corners_reference[i] = T_matrix @ local_corners[i]

        corners_reference = corners_reference[:, :3]
        return corners_reference

    def build_board(self, local_corners):
        visible_markers = []
        corners_dict = {}

        if self.target_poses:
            for pose in self.target_poses:
                pose_values = pose.object_to_reference_matrix.values
                pose_matrix = np.array(pose_values).reshape(4, 4)
                corners_location = self._calculate_corners_location(pose_matrix, local_corners)

                corners_dict[pose.target_id] = corners_location
                visible_markers.append(pose.target_id)

            ### ID IS NOT IN FRAME ###
            for marker_uuid in list(self.marker_id_to_uuid.values()):
                if marker_uuid not in visible_markers:
                    estimated_pose_location = PoseLocation()
                    for other_marker_pose in self.target_poses:
                        matrix_index = self._find_matrix_input_index(other_marker_pose.target_id, marker_uuid, self.marker_id_to_uuid, self.index_to_marker_id)

                        if self.relative_pose_matrix[matrix_index[0]][matrix_index[1]] and other_marker_pose.target_id in visible_markers:
                            T_AB = other_marker_pose.object_to_reference_matrix.values
                            T_AB = np.reshape(T_AB, (4, 4))
                            T_BC = self.relative_pose_matrix[matrix_index[0]][matrix_index[1]].get_TMatrix()
                            T_AC = self._estimate_reference_to_not_visible(T_AB, T_BC)
                            estimated_pose_location.add_matrix(T_AC)
                    marker_pose_matrix = estimated_pose_location.get_TMatrix()
                    invisible_corners_location = self._calculate_corners_location(marker_pose_matrix, local_corners)
                    corners_dict[marker_uuid] = invisible_corners_location

        return corners_dict

