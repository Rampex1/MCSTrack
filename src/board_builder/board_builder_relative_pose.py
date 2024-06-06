import cv2
from cv2 import aruco
import numpy as np
from src.board_builder.pose_location import PoseLocation
from src.pose_solver.pose_solver2 import PoseSolver2

class BoardBuilder:

    def __init__(self):

        ### INIT ###
        self._detector_poses = []
        self._target_poses = []
        self._visible_markers = []
        self._index_counter = 0
        self._index_to_marker_uuid = {}


        ### MATRIX INIT ###
        self._matrix_size = 0
        self._relative_pose_matrix = [[None for _ in range(self._matrix_size)] for _ in range(self._matrix_size)]
        self._local_relative_pose_matrix = []


        ### CORNERS INIT ###
        self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS = 20
        self.local_corners = np.array([
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-left
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-right
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Bottom-right
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1]  # Bottom-left
        ])


    def _calculate_relative_transform(self, T1, T2):
        """ Given transform T1 from reference to marker 1, and transfrom T2 from reference to marker 2, calculate the
        transform from T1 to T2"""
        T1_inv = np.linalg.inv(T1)
        relative_T = np.dot(T1_inv, T2)
        return relative_T

    def _estimate_reference_to_not_visible(self, T_AB, T_BC):
        """ Calculates a marker that is not visible given its relative position from a visible marker"""
        T_AC = np.dot(T_AB, T_BC)
        return T_AC

    def _calculate_corners_location(self, T_matrix, local_corners):
        """ Given a matrix transformation, find the four corners """
        corners_reference = np.zeros((4, 4))
        for i in range(4):
            corners_reference[i] = T_matrix @ local_corners[i]

        corners_reference = corners_reference[:, :3]
        return corners_reference

    def _find_matrix_input_index(self, pose_uuid, other_pose_uuid):
        """ Given two pose uuids, return their index location in the relative pose matrix """

        pose_index = -1
        other_pose_index = -1

        print(self._index_to_marker_uuid)
        for index in self._index_to_marker_uuid:
            if self._index_to_marker_uuid[index] == pose_uuid:
                pose_index = index
            if self._index_to_marker_uuid[index] == other_pose_uuid:
                other_pose_index = index

        return pose_index, other_pose_index

    def _get_occluded_markers_pose(self):
        """ Find the pose of occluded markers based on the pose of visible markers """
        corners_dict = {}

        if self._target_poses:
            for pose in self._target_poses:
                pose_values = pose.object_to_reference_matrix.values
                pose_matrix = np.array(pose_values).reshape(4, 4)
                corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
                corners_dict[pose.target_id] = corners_location

            ### ID IS NOT IN FRAME ###
            for marker_uuid in list(self._index_to_marker_uuid.values()):
                if marker_uuid not in self._visible_markers:
                    estimated_pose_location = PoseLocation()
                    for other_marker_pose in self._target_poses:
                        matrix_index = self._find_matrix_input_index(other_marker_pose.target_id, marker_uuid)

                        if self._relative_pose_matrix[matrix_index[0]][matrix_index[1]] and other_marker_pose.target_id in self._visible_markers:
                            T_AB = other_marker_pose.object_to_reference_matrix.values
                            T_AB = np.reshape(T_AB, (4, 4))
                            T_BC = self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].get_TMatrix()
                            T_AC = self._estimate_reference_to_not_visible(T_AB, T_BC)
                            estimated_pose_location.add_matrix(T_AC)
                    marker_pose_matrix = estimated_pose_location.get_TMatrix()
                    invisible_corners_location = self._calculate_corners_location(marker_pose_matrix, self.local_corners)
                    corners_dict[marker_uuid] = invisible_corners_location

        return corners_dict

    ### PUBLIC METHOD ###
    def expand_matrix(self):
        """ Adds one row and one column to the matrix and initializes them to None """
        size = len(self._relative_pose_matrix) + 1
        new_matrix = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size - 1):
            for j in range(size - 1):
                new_matrix[i][j] = self._relative_pose_matrix[i][j]
        self._relative_pose_matrix = new_matrix
        print(self._relative_pose_matrix)

    def add_detector_poses(self, detector_poses):
        self._detector_poses = detector_poses

    def add_target_poses(self, target_poses):
        visible_markers = []
        self._target_poses = target_poses
        for pose in target_poses:
            visible_markers.append(pose.target_id)
            if pose.target_id not in list(self._index_to_marker_uuid.values()):
                self._index_to_marker_uuid[self._index_counter] = pose.target_id
                self._index_counter += 1

        self._visible_markers = visible_markers


    def collect_data(self):
        """ Collects data of relative position and is entered in matrix"""
        for index, pose in enumerate(self._target_poses):
            # R R R T
            # R R R T
            # R R R T
            # 0 0 0 1


            pose_values = pose.object_to_reference_matrix.values
            pose_matrix = np.array(pose_values).reshape(4, 4)


            for other_pose in self._target_poses:
                if other_pose != pose:
                    other_matrix_values = other_pose.object_to_reference_matrix.values
                    other_pose_matrix = np.array(other_matrix_values).reshape(4, 4)
                    relative_transform = self._calculate_relative_transform(pose_matrix, other_pose_matrix)
                    matrix_index = self._find_matrix_input_index(pose.target_id, other_pose.target_id)

                    if not self._relative_pose_matrix[matrix_index[0]][matrix_index[1]]:
                        new_pose_location = PoseLocation()
                        new_pose_location.add_matrix(relative_transform)
                        self._relative_pose_matrix[matrix_index[0]][matrix_index[1]] = new_pose_location
                    else:
                        self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].add_matrix(relative_transform)

        return self._relative_pose_matrix

    def build_board(self, pose_solver2):
        """ Builds board using the relative matrix"""


        ### GET POSE FROM CAMERA WITHOUT REFERENCE ###
        detector_poses_by_label = dict()
        for pose in self._detector_poses:
            detector_poses_by_label[pose.target_id] = pose.object_to_reference_matrix
        pose_solver2.set_detector_poses(detector_poses_by_label)

        pose_solver2.update()
        _, target_poses = pose_solver2.get_poses()
        self._target_poses = target_poses

        ### FIND THE OCCLUDED MARKERS POSE WITH ALGORITHM ###

        print("relative_pose", self._relative_pose_matrix)
        occluded_markers_dict = self._get_occluded_markers_pose()
        print("Occluded marker dict", occluded_markers_dict)
        return occluded_markers_dict

        # return self._local_relative_pose_matrix
