import datetime
import json
import os
import numpy as np

from collections import defaultdict
from typing import Final

from .utils import BoardBuilderPoseSolver
from .structures import PoseLocation
from src.common.structures import Pose, MarkerSnapshot, MarkerCorners, Matrix4x4
from ..pose_solver.structures import Marker, TargetBoard

_HOMOGENEOUS_POINT_COORD: Final[int] = 4

class BoardBuilder:
    _detector_poses_median: dict[str, PoseLocation]
    _detector_poses: list[Pose]
    _target_poses: list[Pose]
    _index_to_marker_uuid: dict[int, str]
    _index_to_marker_id: dict[int, str]

    _matrix_id_index: int
    _matrix_uuid_index: int
    relative_pose_matrix = list[list[PoseLocation | None]]  # Indexed as [row_index][col_index]
    local_corners = list[list[int]]  # Indexed as [point_index][coordinate_index]

    def __init__(self, marker_size):

        # pose solver init
        self._detector_poses_median = dict()
        self.detector_poses = list()
        self.target_poses = list()
        self.marker_size = marker_size
        # TODO: Could combine the following into something like dict[index, list[id, uuid]] to be more clean
        self._index_to_marker_uuid = dict()
        self._index_to_marker_id = dict()
        self.pose_solver = BoardBuilderPoseSolver()

        # matrix init
        self._matrix_id_index = 0
        self._matrix_uuid_index = 0
        self.relative_pose_matrix = list()
        self.local_corners = [
            [-marker_size / 2, marker_size / 2, 0, 1],  # Top-left
            [marker_size / 2, marker_size / 2, 0, 1],  # Top-right
            [marker_size / 2, -marker_size / 2, 0, 1],  # Bottom-right
            [-marker_size / 2, -marker_size / 2, 0, 1]]  # Bottom-left

        # clear data recording file
        self._clear_data_recording_file()

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

    @staticmethod
    def _clear_data_recording_file():
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data_recording.json')
        file_path = os.path.abspath(file_path)

        # Clear the content of the file
        with open(file_path, "w") as f:
            f.write("{}")

    def _expand_relative_pose_matrix(self):
        """ Adds one row and one column to the matrix and initializes them to None """
        size = len(self.relative_pose_matrix) + 1
        new_matrix = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size - 1):
            for j in range(size - 1):
                new_matrix[i][j] = self.relative_pose_matrix[i][j]
        self.relative_pose_matrix = new_matrix

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
                if marker_snapshot.label not in list(self._index_to_marker_id.values()):
                    self.pose_solver.add_target_marker(int(marker_snapshot.label))
                    self._expand_relative_pose_matrix()
                    self._index_to_marker_id[self._matrix_id_index] = marker_snapshot.label
                    self._matrix_id_index += 1

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
        for pose in target_poses:
            if pose.target_id not in list(self._index_to_marker_uuid.values()):
                self._index_to_marker_uuid[self._matrix_uuid_index] = pose.target_id
                self._matrix_uuid_index += 1

    @staticmethod
    def _write_corners_dict_to_repeatability_test_file(corners_dict):
        corners_dict_serializable = {k: v.tolist() for k, v in corners_dict.items()}

        script_dir = os.path.dirname(os.path.abspath(__file__))
        repeatability_dir = os.path.join(script_dir, 'test', 'repeatability')
        if not os.path.exists(repeatability_dir):
            os.makedirs(repeatability_dir)
        file_path = os.path.join(repeatability_dir, 'board_builder_results.json')

        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []
        data.append(corners_dict_serializable)

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def _write_detector_data_to_file(detector_data: dict[str, list[MarkerSnapshot]], data_description: str):
        formatted_data = {}
        for detector_name, snapshots in detector_data.items():
            formatted_data[detector_name] = []
            for snapshot in snapshots:
                snapshot_data = {
                    "label": snapshot.label,
                    "corner_image_points": [{"x_px": pt.x_px, "y_px": pt.y_px} for pt in snapshot.corner_image_points]
                }
                formatted_data[detector_name].append(snapshot_data)

        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data_recording.json')
        file_path = os.path.abspath(file_path)

        # Read existing data
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        # Determine the next index
        if existing_data:
            next_index = max(map(int, existing_data.keys())) + 1
        else:
            next_index = 1

        # Update existing data with new data
        existing_data[next_index] = [{"name": data_description}, {"formatted_data": formatted_data}]

        # Write updated data back to the file
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)

    # public methods
    def locate_reference_board(self, detector_data: dict[str, list[MarkerSnapshot]]):
        self._write_detector_data_to_file(detector_data, "LOCATE REFERENCE DATA")
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
        self._write_detector_data_to_file(detector_data, "COLLECTION DATA")
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
                    matrix_uuid_index = self._find_matrix_input_index(pose.target_id, other_pose.target_id)

                    if not self.relative_pose_matrix[matrix_uuid_index[0]][matrix_uuid_index[1]]:
                        new_pose_location = PoseLocation(pose.target_id)
                        new_pose_location.add_matrix(relative_transform, timestamp.isoformat())
                        self.relative_pose_matrix[matrix_uuid_index[0]][matrix_uuid_index[1]] = new_pose_location
                    else:
                        self.relative_pose_matrix[matrix_uuid_index[0]][matrix_uuid_index[1]].add_matrix(
                            relative_transform,
                            timestamp.isoformat())
                    self.relative_pose_matrix[matrix_uuid_index[0]][matrix_uuid_index[1]].frame_count += 1

        for index, pose in enumerate(self.target_poses):
            pose_values = pose.object_to_reference_matrix.values
            pose_matrix = np.array(pose_values).reshape(4, 4)
            corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
            corners_dict[pose.target_id] = corners_location

        return corners_dict

    def build_board(self, repeatability_testing=False):
        # TODO: Build board isn't currently able to get the positions of markers with no direct relation with the
        #  reference (appeared in the same frame more than once). A search algorithm needs to be developed

        if not self.relative_pose_matrix:
            return
        self.target_poses = []
        predicted_corners = {}
        relative_pose_matrix = [[entry.get_matrix() if entry is not None else None for entry in row] for
                                row in self.relative_pose_matrix]

        # TODO: Current reference/origin marker is coded to the index with the lowest marker id. An algorithm could be
        #  developed to determine "most reliable marker" to put as the root of the search tree
        index_with_smallest_value = min(self._index_to_marker_id, key=self._index_to_marker_id.get)
        reference_face = self._index_to_marker_id[index_with_smallest_value]
        identity_matrix = np.eye(4)

        for index, marker_id in self._index_to_marker_id.items():
            if marker_id == reference_face:
                # No transformation needed for the reference face
                corners = self._calculate_corners_location(identity_matrix, self.local_corners)
            else:
                # Transformation from face i to reference face
                T = relative_pose_matrix[index_with_smallest_value][index]
                if T is None:
                    continue
                pose = Pose(
                    target_id=marker_id,
                    object_to_reference_matrix=Matrix4x4.from_numpy_array(np.array(T)),
                    solver_timestamp_utc_iso8601=str(datetime.datetime.utcnow()))
                self.target_poses.append(pose)
                corners = self._calculate_corners_location(T, self.local_corners)
            predicted_corners[marker_id] = corners

        if repeatability_testing:
            self._write_corners_dict_to_repeatability_test_file(predicted_corners)

        # Convert to target board
        markers = []
        for marker_id, points in predicted_corners.items():
            # Ensure points is a list of lists
            points = [list(point) for point in points]
            marker = Marker(marker_id=marker_id, points=points)
            markers.append(marker)
        return TargetBoard(target_id='board 1', markers=markers)
