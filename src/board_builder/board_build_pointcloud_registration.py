import cv2
from cv2 import aruco
import numpy as np
import datetime

from typing import Final
from src.common.structures import IntrinsicParameters
from src.pose_solver.pose_solver import PoseSolver
from src.pose_solver.structures import MarkerCorners, TargetMarker, Target
from src.board_builder.pose_location import PoseLocation


class BoardBuilder:
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

        self.target_marker_to_uuid = {}
        self.index_to_target_marker = {}
        self.corners_dict = {}

        # Defines local coordinates from middle to corners. NOTE: the value should be a result of calibration and not hard coded
        self.local_corners = np.array([
            [-17.5, 17.5, 0, 1],  # Top-left
            [17.5, 17.5, 0, 1],   # Top-right
            [17.5, -17.5, 0, 1],  # Bottom-right
            [-17.5, -17.5, 0, 1]  # Bottom-left
        ])


    def _calculate_corners_location(self, T_matrix, local_corners):
        corners_reference = np.zeros((4, 4))
        for i in range(4):
            corners_reference[i] = T_matrix @ local_corners[i]

        corners_reference = corners_reference[:, :3]
        return corners_reference

    def _draw_corners_location(self, corners_location, frame):
        for corner in corners_location:
            x, y, z = corner
            cv2.circle(frame, (int(x) + 200, - int(y) + 200), 5, (0, 0, 255), -1)  # Red color with a radius of 5

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            visible_markers = []  # List of markers that are visible in a specific frame

            ### ADD TARGET MARKER ###
            for marker_id in range(len(ids)):
                visible_markers.append(ids[marker_id][0])
                if ids[marker_id][0] not in self.target_marker_to_uuid and ids[marker_id][
                    0] != self.REFERENCE_MARKER_ID:
                    target_marker_diameter = self.MARKER_SIZE_MM
                    marker_uuid = self.pose_solver.try_add_target_marker(ids[marker_id][0], target_marker_diameter)
                    self.target_marker_to_uuid[ids[marker_id][0]] = marker_uuid

            ### ADD CORNERS ###
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label=self.DETECTOR_GREEN_NAME,
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),
                    timestamp=datetime.datetime.now()
                )
                # print(corner[0].tolist())
                self.pose_solver.add_marker_corners([marker_corners])

            ### SOLVE POSE ###
            self.pose_solver.update()
            detector_poses, target_poses = self.pose_solver.get_poses()

            for pose in target_poses:
                # R R R T
                # R R R T
                # R R R T
                # 0 0 0 1

                # print("POSE", pose.target_id, pose.object_to_reference_matrix.values)

                pose_values = pose.object_to_reference_matrix.values
                pose_matrix = np.array(pose_values).reshape(4, 4)

                corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
                self._draw_corners_location(corners_location, frame)

            ### ID IS NOT IN FRAME ###
            for marker in list(self.target_marker_to_uuid.keys()):
                if marker not in visible_markers:
                    print("NOT VISIBLE")

        return frame

    def run(self):
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_with_markers = self.process_frame(frame)
            cv2.imshow('Frame with ArUco markers', frame_with_markers)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    board_builder = BoardBuilder()
    board_builder.run()
