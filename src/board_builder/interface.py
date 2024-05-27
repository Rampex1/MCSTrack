import cv2
from cv2 import aruco
import numpy as np
import datetime

from typing import Final
from src.common.structures import IntrinsicParameters
from src.pose_solver.pose_solver import PoseSolver
from src.pose_solver.structures import MarkerCorners, TargetMarker, Target
from src.board_builder.pose_location import PoseLocation
from src.board_builder.board_builder_relative_pose import BoardBuilder


class Interface:
    REFERENCE_MARKER_ID: Final[int] = 0
    MARKER_SIZE_MM: Final[float] = 10.0
    DETECTOR_GREEN_NAME: Final[str] = "default_camera"
    DETECTOR_GREEN_INTRINSICS: Final[IntrinsicParameters] = IntrinsicParameters(
        focal_length_x_px=629.7257712407858,
        focal_length_y_px=631.1144336572407,
        optical_center_x_px=327.78473901724755,
        optical_center_y_px=226.74054836282653,
        radial_distortion_coefficients=[
            0.05560270909494751,
            -0.28733139601291297,
            1.182627063988894],
        tangential_distortion_coefficients=[
            -0.00454124371092251,
            0.0009635939551320261])

    def __init__(self):
        self.camera_matrix = np.array([
            [self.DETECTOR_GREEN_INTRINSICS.focal_length_x_px, 0, self.DETECTOR_GREEN_INTRINSICS.optical_center_x_px],
            [0, self.DETECTOR_GREEN_INTRINSICS.focal_length_y_px, self.DETECTOR_GREEN_INTRINSICS.optical_center_y_px],
            [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.array(
            self.DETECTOR_GREEN_INTRINSICS.radial_distortion_coefficients + self.DETECTOR_GREEN_INTRINSICS.tangential_distortion_coefficients)
        self.pose_solver = PoseSolver()
        self.pose_solver.set_intrinsic_parameters(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        reference_target: Target = TargetMarker(
            marker_id=self.REFERENCE_MARKER_ID,
            marker_size=self.MARKER_SIZE_MM)
        self.pose_solver.set_reference_target(reference_target)


        ### COLLECT DATA ###
        self._matrix_size = 0
        self._relative_pose_matrix = [[None for _ in range(self._matrix_size)] for _ in range(self._matrix_size)]
        self._marker_id_to_uuid = {}
        self._index_to_marker_id = {}
        self._target_poses = []
        self._visible_markers = []


        self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS = 20
        self.local_corners = np.array([
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-left
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-right
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Bottom-right
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1]  # Bottom-left
        ])
        self.marker_color = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (128, 0, 0),  # Dark Red
            (0, 128, 0),  # Dark Green
            (0, 0, 128),  # Dark Blue
            (128, 128, 0),  # Olive
            (0, 128, 128),  # Teal
            (128, 0, 128)  # Purple
        ]


    def _expand_matrix(self, matrix):
        """ Adds one row and one column to the matrix and initializes them to None """
        size = len(matrix) + 1
        new_matrix = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size - 1):
            for j in range(size - 1):
                new_matrix[i][j] = matrix[i][j]
        return new_matrix

    def _calculate_relative_transform(self, T1, T2):
        """ Given transform T1 from reference to marker 1, and transfrom T2 from reference to marker 2, calculate the
        transform from T1 to T2"""
        T1_inv = np.linalg.inv(T1)
        relative_T = np.dot(T1_inv, T2)
        return relative_T

    def _find_matrix_input_index(self, pose_uuid, other_pose_uuid):
        """ Given two pose uuids, return their index location in the relative pose matrix """
        pose_id = -1
        other_pose_id = -1
        pose_index = -1
        other_pose_index = -1

        for id in self._marker_id_to_uuid:
            if self._marker_id_to_uuid[id] == pose_uuid:
                pose_id = id
            if self._marker_id_to_uuid[id] == other_pose_uuid:
                other_pose_id = id

        if pose_id != -1 and other_pose_id != -1:
            for index in self._index_to_marker_id:
                if self._index_to_marker_id[index] == pose_id:
                    pose_index = index
                if self._index_to_marker_id[index] == other_pose_id:
                    other_pose_index = index
            return pose_index, other_pose_index

        return None

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

    def _draw_corners_location(self, corners_location, frame, marker_color):
        """ Takes in a list of three coordinates (x,y,z) and draws it on the board with different colors every 4 iterations."""

        for corner in corners_location:
            x, y, z = corner
            cv2.circle(frame, (int(x) + 200, - int(y) + 200), 4, marker_color, -1)

    def _draw_board_builder(self, frame):
        corners_dict = {}

        if self._target_poses:
            for pose in self._target_poses:
                pose_values = pose.object_to_reference_matrix.values
                pose_matrix = np.array(pose_values).reshape(4, 4)
                corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)

                corners_dict[pose.target_id] = corners_location
                self._visible_markers.append(pose.target_id)

            ### ID IS NOT IN FRAME ###
            for marker_uuid in list(self._marker_id_to_uuid.values()):
                if marker_uuid not in self._visible_markers:
                    estimated_pose_location = PoseLocation()
                    for other_marker_pose in self._target_poses:
                        matrix_index = self._find_matrix_input_index(other_marker_pose.target_id, marker_uuid)

                        if self._relative_pose_matrix[matrix_index[0]][
                            matrix_index[1]] and other_marker_pose.target_id in self._visible_markers:
                            T_AB = other_marker_pose.object_to_reference_matrix.values
                            T_AB = np.reshape(T_AB, (4, 4))
                            T_BC = self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].get_TMatrix()
                            T_AC = self._estimate_reference_to_not_visible(T_AB, T_BC)
                            estimated_pose_location.add_matrix(T_AC)
                    marker_pose_matrix = estimated_pose_location.get_TMatrix()
                    invisible_corners_location = self._calculate_corners_location(marker_pose_matrix,
                                                                                  self.local_corners)
                    corners_dict[marker_uuid] = invisible_corners_location

        for index, marker_uuid in enumerate(corners_dict):
            color_index = index % len(self.marker_color)
            self._draw_corners_location(corners_dict[marker_uuid], frame, self.marker_color[color_index])

    def _solve_pose(self, ids, corners):
        """ Given visible Ids and their corners, uses pose_solver to solve the pose """
        if ids is not None:
            visible_markers = []  # List of markers that are visible in a specific frame

            ### ADD TARGET MARKER ###
            for marker_id in range(len(ids)):
                visible_markers.append(ids[marker_id][0])
                if ids[marker_id][0] not in self._marker_id_to_uuid and ids[marker_id][
                    0] != self.REFERENCE_MARKER_ID:
                    marker_uuid = self.pose_solver.try_add_target_marker(ids[marker_id][0], int(self.MARKER_SIZE_MM))
                    self._marker_id_to_uuid[ids[marker_id][0]] = marker_uuid
                    self._index_to_marker_id[self._matrix_size] = ids[marker_id][0]

                    ### EXPAND MATRIX ###
                    self._relative_pose_matrix = self._expand_matrix(self._relative_pose_matrix)
                    self._matrix_size += 1
            self._visible_markers = visible_markers

            ### ADD CORNERS ###
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label=self.DETECTOR_GREEN_NAME,
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),
                    timestamp=datetime.datetime.now()
                )
                self.pose_solver.add_marker_corners([marker_corners])

            ### SOLVE POSE ###
            self.pose_solver.update()
            detector_poses, target_poses = self.pose_solver.get_poses()
            self._target_poses = target_poses

    def _collect_data(self, frame):
        """ Collects data of relative position and is entered in matrix"""
        for index, pose in enumerate(self._target_poses):
            # R R R T
            # R R R T
            # R R R T
            # 0 0 0 1
            #print("POSE", pose.target_id, pose.object_to_reference_matrix.values)


            pose_values = pose.object_to_reference_matrix.values
            pose_matrix = np.array(pose_values).reshape(4, 4)

            color_index = index % len(self.marker_color)
            corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
            self._draw_corners_location(corners_location, frame, self.marker_color[color_index])

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

    def _build_board(self, frame):
        board_builder = BoardBuilder(self._relative_pose_matrix, self._marker_id_to_uuid, self._index_to_marker_id)
        board_builder.build_board()

        self._draw_board_builder(frame)


    def update(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        collecting_data = False
        building_board = False

        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            aruco.drawDetectedMarkers(frame, corners, ids)

            if not ret:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                collecting_data = True
                building_board = False
            elif key == ord('b'):
                collecting_data = False
                building_board = True
            elif key == ord('r'):
                collecting_data = False
                building_board = False
                self._matrix_size = 0
                self._relative_pose_matrix = [[None for _ in range(self._matrix_size)] for _ in
                                              range(self._matrix_size)]
                self._marker_id_to_uuid = {}
                self._index_to_marker_id = {}
                self._target_poses = []
                self._visible_markers = []


                self.pose_solver = PoseSolver()
                self.pose_solver.set_intrinsic_parameters(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
                reference_target: Target = TargetMarker(
                    marker_id=self.REFERENCE_MARKER_ID,
                    marker_size=self.MARKER_SIZE_MM)
                self.pose_solver.set_reference_target(reference_target)

            elif key == ord('q'):
                break

            if collecting_data:
                self._solve_pose(ids, corners)
                self._collect_data(frame)
            elif building_board:
                self._solve_pose(ids, corners)
                self._build_board(frame)

            cv2.imshow('Frame', frame)



        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    board_builder = Interface()
    board_builder.update()
