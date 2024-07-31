import numpy as np
import datetime
from .utils.board_builder_pose_solver import BoardBuilderPoseSolver
from .structures.pose_location import PoseLocation
from src.common.structures import Pose, MarkerSnapshot, MarkerCorners, Matrix4x4
from collections import defaultdict
from typing import Final


_HOMOGENEOUS_POINT_COORD: Final[int] = 4
_MARKER_MIDDLE_TO_EDGE_IN_PIXELS: Final[float] = 20.0


class BoardBuilder:
    _detector_poses_median: dict[str, PoseLocation]
    _detector_poses: list[Pose]
    _target_poses: list[Pose]
    _visible_markers: list[str]
    _target_markers: list[str]
    _index_to_marker_uuid: dict[int, str]

    _relative_pose_matrix = list[list[PoseLocation | None]]  # Indexed as [row_index][col_index]
    _local_corners = list[list[int]]  # Indexed as [point_index][coordinate_index]

    def __init__(self):

        # pose solver init
        self._detector_poses_median = dict()
        self.detector_poses = list()
        self.target_poses = list()
        self._visible_markers = list()
        self._index_to_marker_uuid = dict()
        self._target_markers = list()
        self.pose_solver = BoardBuilderPoseSolver()

        # matrix init
        self._matrix_index = 0
        self._relative_pose_matrix = list()
        # TODO: Will need to be calculated rather than hard coded
        self.local_corners = [
            [-_MARKER_MIDDLE_TO_EDGE_IN_PIXELS, _MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-left
            [_MARKER_MIDDLE_TO_EDGE_IN_PIXELS, _MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-right
            [_MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -_MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Bottom-right
            [-_MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -_MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1]]  # Bottom-left

    @staticmethod
    def _calculate_corners_location(
        t_matrix,
        local_corners: list[list[int]]
    ):
        """
        Given a matrix transformation, find the location of the corners.

        Args:
            t_matrix (np.ndarray): Transformation matrix (4x4).
            local_corners (List[List[int]]): Local corners coordinates, expected to be 4 points with 4 coordinates each.

        Returns:
            np.ndarray: Transformed corners coordinates (Nx3).
        """

        corners_reference = np.zeros((_HOMOGENEOUS_POINT_COORD, _HOMOGENEOUS_POINT_COORD))
        for i in range(len(local_corners)):
            corners_reference[i] = t_matrix @ np.asarray(local_corners[i])

        corners_reference = corners_reference[:, :3]
        return corners_reference

    @staticmethod
    def _calculate_relative_transform(t1, t2):
        """ Given transform T1 from reference to marker 1, and transfrom T2 from reference to marker 2, calculate the
        transform from T1 to T2"""
        t1_inv = np.linalg.inv(t1)
        t_relative = t1_inv @ t2
        return t_relative

    def _expand_relative_pose_matrix(self):
        """ Adds one row and one column to the matrix and initializes them to None """
        size = len(self._relative_pose_matrix) + 1
        new_matrix = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size - 1):
            for j in range(size - 1):
                new_matrix[i][j] = self._relative_pose_matrix[i][j]
        self._relative_pose_matrix = new_matrix

    @staticmethod
    def _filter_markers_appearing_in_multiple_detectors(data):
        """ Filters the detector data to only keep the markers snapshots that appear in more than one detector """
        label_counts = defaultdict(int)
        for snapshots in data.values():
            for snapshot in snapshots:
                label_counts[snapshot.label] += 1

        filtered_data = {}
        for key, snapshots in data.items():
            filtered_snapshots = [
                snapshot for snapshot in snapshots
                if label_counts[snapshot.label] >= 2]
            if filtered_snapshots:
                filtered_data[key] = filtered_snapshots
        return filtered_data

    def _find_matrix_input_index(self, pose_uuid, other_pose_uuid):
        """ Given two pose uuids, return their index location in the relative pose matrix """

        pose_index = -1
        other_pose_index = -1

        for index in self._index_to_marker_uuid:
            if self._index_to_marker_uuid[index] == pose_uuid:
                pose_index = index
            if self._index_to_marker_uuid[index] == other_pose_uuid:
                other_pose_index = index

        if pose_index == -1:
            raise RuntimeError(f"Pose UUID {pose_uuid} not found in the index mapping.")
        if other_pose_index == -1:
            raise RuntimeError(f"Pose UUID {other_pose_uuid} not found in the index mapping.")

        return pose_index, other_pose_index

    def _solve_pose(self, detector_data: dict[str, list[MarkerSnapshot]], timestamp: datetime.datetime):
        """ Given marker ids and its corner locations, find its pose """
        timestamp = datetime.datetime.utcnow()
        for detector_name in detector_data:
            for marker_snapshot in detector_data[detector_name]:
                if marker_snapshot.label not in self._target_markers:
                    self._target_markers.append(marker_snapshot.label)
                    self.pose_solver.add_target_marker(int(marker_snapshot.label))
                    self._expand_relative_pose_matrix()

        for detector_name in detector_data:
            for marker_snapshot in detector_data[detector_name]:
                corners_list: list[list[float]] = []  # Indexed as [point][coordinate]
                for corner in marker_snapshot.corner_image_points:
                    corners_list.append([corner.x_px, corner.y_px])
                marker_corners = MarkerCorners(
                    detector_label=detector_name,
                    marker_id=int(marker_snapshot.label),
                    points=corners_list,
                    timestamp=timestamp
                )
                self.pose_solver.add_marker_corners([marker_corners])

        target_poses = self.pose_solver.get_target_poses()
        self.target_poses = target_poses
        visible_markers = []
        for pose in target_poses:
            visible_markers.append(pose.target_id)
            if pose.target_id not in list(self._index_to_marker_uuid.values()):
                self._index_to_marker_uuid[self._matrix_index] = pose.target_id
                self._matrix_index += 1
        self._visible_markers = visible_markers

    # public methods
    def locate_reference_board(self, detector_data: dict[str, list[MarkerSnapshot]]):
        if all(isinstance(v, list) and len(v) == 0 for v in detector_data.values()):
            return
        self.detector_poses = []
        timestamp = datetime.datetime.utcnow()
        for detector_name in detector_data:
            for marker_snapshot in detector_data[detector_name]:
                corners_list: list[list[float]] = []
                for corner in marker_snapshot.corner_image_points:
                    corners_list.append([corner.x_px, corner.y_px])
                marker_corners = MarkerCorners(
                    detector_label=detector_name,
                    marker_id=int(marker_snapshot.label),
                    points=corners_list,
                    timestamp=timestamp
                )
                self.pose_solver.add_marker_corners([marker_corners])

        new_detector_poses = self.pose_solver.get_detector_poses()
        for pose in new_detector_poses:
            if pose.target_id not in self._detector_poses_median:
                self._detector_poses_median[pose.target_id] = PoseLocation(pose.target_id)
            self._detector_poses_median[pose.target_id].add_matrix(
                pose.object_to_reference_matrix.as_numpy_array(),
                timestamp.isoformat())
        for label in self._detector_poses_median:
            pose = Pose(
                target_id=label,
                object_to_reference_matrix=self._detector_poses_median[label].get_median_pose().object_to_reference_matrix,
                solver_timestamp_utc_iso8601=timestamp.isoformat())
            self.detector_poses.append(pose)
        self.pose_solver.set_detector_poses(self.detector_poses)

    def collect_data(self, detector_data: dict[str, list[MarkerSnapshot]]):
        """ Collects data of relative position and is entered in matrix. Returns a dictionary of its corners"""
        detector_data = self._filter_markers_appearing_in_multiple_detectors(detector_data)
        if all(isinstance(v, list) and len(v) == 0 for v in detector_data.values()):
            return
        timestamp = datetime.datetime.utcnow()
        corners_dict = {}
        self.target_poses = []
        self._solve_pose(detector_data, timestamp)
        for index, pose in enumerate(self.target_poses):
            # R R R T
            # R R R T
            # R R R T
            # 0 0 0 1

            pose_values = pose.object_to_reference_matrix.values
            pose_matrix = np.array(pose_values).reshape(4, 4)

            for other_pose in self.target_poses:
                if other_pose != pose:
                    other_matrix_values = other_pose.object_to_reference_matrix.values
                    other_pose_matrix = np.array(other_matrix_values).reshape(4, 4)
                    relative_transform = self._calculate_relative_transform(pose_matrix, other_pose_matrix)
                    matrix_index = self._find_matrix_input_index(pose.target_id, other_pose.target_id)

                    if not self._relative_pose_matrix[matrix_index[0]][matrix_index[1]]:
                        new_pose_location = PoseLocation(pose.target_id)
                        new_pose_location.add_matrix(relative_transform, timestamp.isoformat())
                        self._relative_pose_matrix[matrix_index[0]][matrix_index[1]] = new_pose_location
                    else:
                        self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].add_matrix(
                            relative_transform,
                            timestamp.isoformat())
                    self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].frame_count += 1

        for index, pose in enumerate(self.target_poses):
            pose_values = pose.object_to_reference_matrix.values
            pose_matrix = np.array(pose_values).reshape(4, 4)
            corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
            corners_dict[pose.target_id] = corners_location
        return corners_dict

    def build_board(self):
        predicted_corners = {}
        self.target_poses = []
        relative_pose_matrix = [[entry.get_matrix() if entry is not None else None for entry in row] for
                                row in self._relative_pose_matrix]

        # TODO: Current reference/origin marker is hard coded to the first marker of the matrix
        reference_face = 0
        identity_matrix = np.eye(4)

        for i in range(len(relative_pose_matrix)):
            if i == reference_face:
                # No transformation needed for the reference face
                corners = self._calculate_corners_location(identity_matrix, self.local_corners)
            else:
                # Transformation from face i to reference face
                T = relative_pose_matrix[reference_face][i]
                if T is None:
                    continue
                pose = Pose(
                    target_id=i,
                    object_to_reference_matrix=Matrix4x4.from_numpy_array(np.array(T)),
                    solver_timestamp_utc_iso8601=str(datetime.datetime.utcnow()))
                self.target_poses.append(pose)
                corners = self._calculate_corners_location(T, self.local_corners)
            predicted_corners[i] = corners

        return predicted_corners
