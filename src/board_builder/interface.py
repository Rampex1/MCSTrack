import cv2
from cv2 import aruco
import numpy as np
import datetime

from typing import Final
from src.common.structures import IntrinsicParameters
from src.pose_solver.pose_solver import PoseSolver
from src.pose_solver.structures import MarkerCorners, TargetMarker, Target
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

        ### CAMERA SETTINGS ###
        self.camera_matrix = np.array([
            [self.DETECTOR_GREEN_INTRINSICS.focal_length_x_px, 0, self.DETECTOR_GREEN_INTRINSICS.optical_center_x_px],
            [0, self.DETECTOR_GREEN_INTRINSICS.focal_length_y_px, self.DETECTOR_GREEN_INTRINSICS.optical_center_y_px],
            [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.array(self.DETECTOR_GREEN_INTRINSICS.radial_distortion_coefficients
                                    + self.DETECTOR_GREEN_INTRINSICS.tangential_distortion_coefficients)


        ### POSE SOLVER INIT ###
        self.pose_solver = PoseSolver()
        self.pose_solver.set_intrinsic_parameters(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        self.pose_solver.set_reference_target(TargetMarker(
            marker_id=self.REFERENCE_MARKER_ID,
            marker_size=self.MARKER_SIZE_MM))
        self._target_poses = []


        ### BOARD BUILDER INIT ###
        self.board_builder = BoardBuilder()
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
        ]


    def _solve_pose(self, ids, corners):
        """ Given visible Ids and their corners, uses pose_solver to return the pose in the reference frame """

        if ids is not None:
            ### ADD TARGET MARKER ###
            for marker_id in range(len(ids)):
                if ids[marker_id][0] != self.REFERENCE_MARKER_ID:
                    self.pose_solver.try_add_target_marker(ids[marker_id][0], int(self.MARKER_SIZE_MM))
                    self.board_builder.expand_matrix()

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
                self._target_poses = []
                self.board_builder = BoardBuilder()
            elif key == ord('q'):
                break


            if collecting_data:
                self._solve_pose(ids, corners)

                if self._target_poses is not None:
                    self.board_builder.add_target_poses(self._target_poses)
                    self.board_builder.collect_data()


                    ### DRAW MARKERS ###
                    for index, pose in enumerate(self._target_poses):
                        pose_values = pose.object_to_reference_matrix.values
                        pose_matrix = np.array(pose_values).reshape(4, 4)
                        color_index = index % len(self.marker_color)
                        corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
                        self._draw_corners_location(corners_location, frame, self.marker_color[color_index])

            elif building_board:
                self._solve_pose(ids, corners)

                if self._target_poses is not None:
                    self.board_builder.add_target_poses(self._target_poses)
                    corners_dict = self.board_builder.build_board()
                    for index, marker_uuid in enumerate(corners_dict):
                        color_index = index % len(self.marker_color)
                        self._draw_corners_location(corners_dict[marker_uuid], frame, self.marker_color[color_index])

            cv2.imshow('Frame', frame)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    board_builder = Interface()
    board_builder.update()
