import datetime
import numpy as np
from src.board_builder.utils.board_builder_pose_solver import BoardBuilderPoseSolver
from src.board_builder.structures.pose_location import PoseLocation
from src.common.structures import Pose
from src.pose_solver.structures import MarkerCorners


class BoardBuilder:
    _detector_poses_average: dict[str, PoseLocation]
    _detector_poses: list[Pose]
    _target_poses: list[Pose]
    _occluded_poses: list[Pose]
    _visible_markers: list[str]
    _index_to_marker_uuid: dict[int, str]

    _relative_pose_matrix = list[list[int]]
    _local_corners = list[list[int]]

    def __init__(self):

        ### POSE SOLVER INIT ###
        self._detector_poses_average = {}
        self.detector_poses = []
        self.target_poses = []
        self.occluded_poses = []
        self._visible_markers = []
        self._index_to_marker_uuid = {}
        self.pose_solver = BoardBuilderPoseSolver()

        ### MATRIX INIT ###
        self._matrix_index = 0
        self._relative_pose_matrix = [[None]]
        # TODO: Will need to be calculated rather than hard coded
        self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS = 20
        self.local_corners = np.array([
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-left
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-right
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Bottom-right
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1]  # Bottom-left
        ])

    def _calculate_corners_location(self, T_matrix, local_corners):
        """ Given a matrix transformation, find the location of the four corners """
        corners_reference = np.zeros((4, 4))
        for i in range(4):
            corners_reference[i] = T_matrix @ local_corners[i]

        corners_reference = corners_reference[:, :3]
        return corners_reference

    def _calculate_relative_transform(self, T1, T2):
        """ Given transform T1 from reference to marker 1, and transfrom T2 from reference to marker 2, calculate the
        transform from T1 to T2"""
        T1_inv = np.linalg.inv(T1)
        T_relative = np.dot(T1_inv, T2)
        return T_relative

    def _estimate_reference_to_occluded(self, T1, T2):
        """ Given transform T1 from reference to visible marker, and transform T2 from visible marker to occluded
        marker, calculate the transform from the reference to the occluded marker"""
        T_reference_to_occluded = np.dot(T1, T2)
        return T_reference_to_occluded

    def _expand_matrix(self):
        """ Adds one row and one column to the matrix and initializes them to None """
        size = len(self._relative_pose_matrix) + 1
        new_matrix = [[None for _ in range(size)] for _ in range(size)]
        for i in range(size - 1):
            for j in range(size - 1):
                new_matrix[i][j] = self._relative_pose_matrix[i][j]
        self._relative_pose_matrix = new_matrix

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
            raise ValueError(f"Pose UUID {pose_uuid} not found in the index mapping.")
        if other_pose_index == -1:
            raise ValueError(f"Pose UUID {other_pose_uuid} not found in the index mapping.")

        return pose_index, other_pose_index

    def _solve_pose(self, detector_data):
        """ Given marker ids and its corner locations, find its pose """
        markers_visible = False

        for detector_name in detector_data:
            if detector_data[detector_name]['ids'] is not None:
                markers_visible = True

        if markers_visible:
            for detector_name in detector_data:
                for marker_id in detector_data[detector_name]['ids']:
                    if self.pose_solver.add_target_marker(marker_id):
                        self._expand_matrix()

            for detector_name in detector_data:
                for index, corner in enumerate(detector_data[detector_name]['corners']):
                    marker_corners = MarkerCorners(
                        detector_label=detector_name,
                        marker_id=int(detector_data[detector_name]['ids'][index]),
                        points=corner.tolist(),
                        timestamp=datetime.datetime.now()
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


    ### PUBLIC METHOD ###
    def locate_reference_markers(self, detector_data):

        markers_visible = False
        for detector_name in detector_data:
            if detector_data[detector_name]['ids'] is not None:
                markers_visible = True

        if markers_visible:
            self.detector_poses = []
            for detector_name in detector_data:
                for index, corner in enumerate(detector_data[detector_name]['corners']):
                    marker_corners = MarkerCorners(
                        detector_label=detector_name,
                        marker_id=int(detector_data[detector_name]['ids'][index]),
                        points=corner.tolist(),
                        timestamp=datetime.datetime.now()
                    )
                    self.pose_solver.add_marker_corners([marker_corners])

            new_detector_poses = self.pose_solver.get_detector_poses()
            for pose in new_detector_poses:
                if pose.target_id not in self._detector_poses_average:
                    self._detector_poses_average[pose.target_id] = PoseLocation(pose.target_id)
                self._detector_poses_average[pose.target_id].add_matrix(np.array(pose.object_to_reference_matrix.values).reshape(4, 4), str(datetime.datetime.now()))
            for label in self._detector_poses_average:
                pose = Pose(
                    target_id=label,
                    object_to_reference_matrix=self._detector_poses_average[label].get_pose().object_to_reference_matrix,
                    solver_timestamp_utc_iso8601=str(datetime.datetime.now())
                )
                self.detector_poses.append(pose)
            self.pose_solver.set_detector_poses(self.detector_poses)

    def collect_data(self, detector_data):
        """ Collects data of relative position and is entered in matrix. Returns a dictionary of its corners"""
        corners_dict = {}
        self.target_poses = []
        self._solve_pose(detector_data)
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
                        new_pose_location.add_matrix(relative_transform, str(datetime.datetime.now()))
                        self._relative_pose_matrix[matrix_index[0]][matrix_index[1]] = new_pose_location
                    else:
                        self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].add_matrix(relative_transform, str(datetime.datetime.now()))

        for index, pose in enumerate(self.target_poses):
            pose_values = pose.object_to_reference_matrix.values
            pose_matrix = np.array(pose_values).reshape(4, 4)
            corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
            corners_dict[pose.target_id] = corners_location

        return corners_dict

    def build_board(self, detector_data):
        """ Builds board using the relative matrix"""

        corners_dict = {}
        self.occluded_poses = []
        self._solve_pose(detector_data)
        if self.target_poses:
            for pose in self.target_poses:
                pose_values = pose.object_to_reference_matrix.values
                pose_matrix = np.array(pose_values).reshape(4, 4)
                corners_location = self._calculate_corners_location(pose_matrix, self.local_corners)
                corners_dict[pose.target_id] = corners_location

            ### ID IS NOT IN FRAME ###
            for marker_uuid in list(self._index_to_marker_uuid.values()):
                if marker_uuid not in self._visible_markers:
                    estimated_pose_location = PoseLocation(marker_uuid)
                    for other_marker_pose in self.target_poses:
                        matrix_index = self._find_matrix_input_index(other_marker_pose.target_id, marker_uuid)

                        if (self._relative_pose_matrix[matrix_index[0]][matrix_index[1]] and other_marker_pose.target_id
                                in self._visible_markers):
                            T_AB = other_marker_pose.object_to_reference_matrix.values
                            T_AB = np.reshape(T_AB, (4, 4))
                            T_BC = self._relative_pose_matrix[matrix_index[0]][matrix_index[1]].get_matrix()
                            T_AC = self._estimate_reference_to_occluded(T_AB, T_BC)
                            estimated_pose_location.add_matrix(T_AC, str(datetime.datetime.now()))
                    marker_pose_matrix = estimated_pose_location.get_matrix()
                    pose = estimated_pose_location.get_pose()
                    invisible_corners_location = self._calculate_corners_location(marker_pose_matrix,
                                                                                  self.local_corners)
                    corners_dict[marker_uuid] = invisible_corners_location
                    self.occluded_poses.append(Pose(
                        target_id=pose.target_id,
                        object_to_reference_matrix=pose.object_to_reference_matrix,
                        solver_timestamp_utc_iso8601=pose.solver_timestamp_utc_iso8601
                    ))

        return corners_dict
