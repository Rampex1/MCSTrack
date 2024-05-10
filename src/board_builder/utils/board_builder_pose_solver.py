from src.board_builder.structures import PoseLocation, CharucoBoard
from src.common.structures import \
    IntrinsicParameters, \
    Matrix4x4, \
    Pose
from src.pose_solver.structures import \
    MarkerCorners, \
    MarkerRaySet, \
    PoseData, \
    Ray, \
    Target, \
    TargetMarker, \
    PoseSolverParameters
from src.pose_solver.util import \
    average_quaternion, \
    transformation_image_to_opengl, \
    convex_quadrilateral_area, \
    closest_intersection_between_n_lines, \
    IterativeClosestPointParameters, \
    iterative_closest_point_for_points_and_rays, \
    register_corresponding_points, \
    vector_image_to_opengl
import cv2
import cv2.aruco
import datetime
import numpy
from scipy.spatial.transform import Rotation
from typing import Callable, TypeVar
import uuid

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


class ImagePointSetsKey:
    detector_label: str
    timestamp: datetime.datetime

    def __init__(
            self,
            detector_label: str,
            timestamp: datetime.datetime
    ):
        self.detector_label = detector_label
        self.timestamp = timestamp

    def _key(self):
        return self.detector_label, self.timestamp

    def __eq__(self, other):
        if isinstance(other, ImagePointSetsKey):
            return self._key() == other._key()
        return False

    def __hash__(self):
        return hash(self._key())


class MarkerKey:
    detector_label: str
    marker_id: int

    def __init__(self, detector_label, marker_id):
        self.detector_label = detector_label
        self.marker_id = marker_id

    def _key(self):
        return self.detector_label, self.marker_id

    def __eq__(self, other):
        if isinstance(other, MarkerKey):
            return self._key() == other._key()
        return False

    def __hash__(self):
        return hash(self._key())

    def __str__(self):
        return str("(" + self.detector_label + "," + str(self.marker_id) + ")")


class CornerSetReference:
    marker_id: int
    corners: list[list[float]]  # in reference coordinate system
    ray_sets: list[MarkerRaySet]

    def __init__(
        self,
        marker_id: int,
        corners: list[list[float]],
        ray_sets: list[MarkerRaySet]
    ):
        self.marker_id = marker_id
        self.corners = corners
        self.ray_sets = ray_sets


class TargetDepthKey:
    target_id: uuid.UUID
    detector_label: str

    def __init__(
        self,
        target_id: uuid.UUID,
        detector_label: str
    ):
        self.target_id = target_id
        self.detector_label = detector_label

    def _key(self):
        return self.target_id, self.detector_label

    def __eq__(self, other):
        if isinstance(other, TargetDepthKey):
            return self._key() == other._key()
        return False

    def __hash__(self):
        return hash(self._key())


class TargetDepth:
    target_id: uuid.UUID
    detector_label: str
    estimate_timestamp: datetime.datetime
    depth: float

    def __init__(
        self,
        target_id: uuid.UUID,
        detector_label: str,
        estimate_timestamp: datetime.datetime,
        depth: float
    ):
        self.target_id = target_id
        self.detector_label = detector_label
        self.estimate_timestamp = estimate_timestamp
        self.depth = depth

    @staticmethod
    def age_seconds(
        target_depth,
        query_timestamp: datetime.datetime
    ):
        return (query_timestamp - target_depth.estimate_timestamp).total_seconds()


class BoardBuilderPoseSolver:
    """
    Class containing the actual "solver" logic, kept separate from the API.
    """

    _intrinsics_by_detector_label: dict[str, IntrinsicParameters]
    _targets: dict[uuid.UUID, Target]

    _marker_corners_since_update: list[MarkerCorners]

    _marker_rayset_by_marker_key: dict[MarkerKey, MarkerRaySet]

    _alpha_poses_by_target_id: dict[uuid.UUID, list[PoseData]]
    _target_extrapolation_poses_by_target_id: dict[uuid.UUID, list[PoseData]]
    _poses_by_target_id: dict[uuid.UUID, PoseData]
    _poses_by_detector_label: dict[str, Matrix4x4]
    _target_depths_by_target_depth_key: dict[TargetDepthKey, list[TargetDepth]]
    _poses_average_by_detector_label: dict[str, PoseLocation]
    _detector_poses: list[Pose]

    _minimum_marker_age_before_removal_seconds: float

    _board_marker_ids: list[int]
    _board_marker_positions: list[list[int]]
    _board_marker_size: int

    def __init__(self):

        self._intrinsics_by_detector_label = dict()
        self._parameters = PoseSolverParameters()
        self._targets = dict()
        self._marker_corners_since_update = list()
        self._marker_rayset_by_marker_key = dict()
        self._alpha_poses_by_target_id = dict()
        self._target_extrapolation_poses_by_target_id = dict()
        self._poses_by_target_id = dict()
        self._poses_by_detector_label = dict()
        self._target_depths_by_target_depth_key = dict()
        self._poses_average_by_detector_label = dict()
        self._detector_poses = list()

        self._minimum_marker_age_before_removal_seconds = max([
            self._parameters.POSE_DETECTOR_DENOISE_LIMIT_AGE_SECONDS,
            self._parameters.POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS,
            self._parameters.POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS,
            self._parameters.POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS,
            self._parameters.POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS])

        self._charuco_board = CharucoBoard()
        self._board_marker_ids = self._charuco_board.get_ids()
        self._board_marker_positions = self._charuco_board.get_positions()
        self._board_marker_size = 10

    def add_marker_corners(
        self,
        detected_corners: list[MarkerCorners]
    ) -> None:
        self._marker_corners_since_update += detected_corners

    def add_target_marker(
            self,
            marker_id: int,
    ) -> bool:
        for target_id, target in self._targets.items():
            if isinstance(target, TargetMarker) and marker_id == target.marker_id:
                return False
        target: Target = TargetMarker(
            marker_id=marker_id,
            marker_size=self._board_marker_size)
        target_id: uuid.UUID = uuid.uuid4()
        self._targets[target_id] = target
        return True

    def get_detector_poses(
        self
    ) -> list[Pose]:
        self._estimate_detector_pose_relative_to_reference()
        detector_poses: list[Pose] = [
            Pose(
                target_id=detector_label,
                object_to_reference_matrix=pose,
                solver_timestamp_utc_iso8601=str(datetime.datetime.utcnow().isoformat()))
            for detector_label, pose in self._poses_by_detector_label.items()]
        return detector_poses

    def get_target_poses(
        self
    ) -> list[Pose]:
        self._estimate_target_pose_relative_to_reference()
        target_poses: list[Pose] = [
            Pose(
                target_id=str(target_id),
                object_to_reference_matrix=pose.object_to_reference_matrix,
                solver_timestamp_utc_iso8601=str(pose.newest_timestamp().isoformat()))
            for target_id, pose in self._poses_by_target_id.items()]
        return target_poses

    def set_intrinsic_parameters(
        self,
        detector_label: str,
        intrinsic_parameters: IntrinsicParameters
    ) -> None:
        self._intrinsics_by_detector_label[detector_label] = intrinsic_parameters

    def set_detector_poses(self, detector_poses):
        self._detector_poses = detector_poses

    def set_board_marker_size(self, board_marker_size):
        self._board_marker_size = board_marker_size

    def _calculate_marker_ray_set(
        self,
        image_point_set: MarkerCorners,
        detector_to_reference_matrix: Matrix4x4
    ) -> MarkerRaySet:
        undistorted_points_original: numpy.ndarray = numpy.array(image_point_set.points, dtype="float32")
        undistorted_points_original = numpy.reshape(
            a=undistorted_points_original,
            newshape=(1, len(image_point_set.points), 2))
        camera_matrix: numpy.ndarray = numpy.array(
            self._intrinsics_by_detector_label[image_point_set.detector_label].get_matrix())
        distortion_coefficients: numpy.ndarray = numpy.array(
            self._intrinsics_by_detector_label[image_point_set.detector_label].get_distortion_coefficients())
        undistorted_points_normalized = cv2.undistortPoints(
            src=undistorted_points_original,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients)
        ray_directions: list[list[float]] = list()
        origin_point_detector = [0, 0, 0, 1]  # origin
        detector_to_reference: numpy.ndarray = detector_to_reference_matrix.as_numpy_array()
        ray_origin_reference = numpy.matmul(detector_to_reference, origin_point_detector)
        ray_origin_reference = ray_origin_reference.tolist()[0:3]
        for point_normalized in undistorted_points_normalized:
            target_point_image = [point_normalized[0, 0], point_normalized[0, 1], 1, 1]  # reverse perspective
            target_point_detector = vector_image_to_opengl(target_point_image)
            ray_direction_detector = numpy.subtract(target_point_detector, origin_point_detector)
            ray_direction_detector = ray_direction_detector / numpy.linalg.norm(ray_direction_detector)  # normalize
            ray_direction_reference = numpy.matmul(detector_to_reference, ray_direction_detector)
            ray_directions.append(list(ray_direction_reference[0:3]))
        return MarkerRaySet(
            marker_id=image_point_set.marker_id,
            image_points=image_point_set.points,
            image_timestamp=image_point_set.timestamp,
            detector_label=image_point_set.detector_label,
            detector_to_reference_matrix=detector_to_reference_matrix,
            ray_origin_reference=ray_origin_reference,
            ray_directions_reference=ray_directions)

    def _clear_old_values(
        self,
        query_timestamp: datetime.datetime
    ) -> bool:  # whether any dict's have changed or not
        changed = False
        self._marker_rayset_by_marker_key, modified = self._clear_old_values_from_dict(
            input_dict=self._marker_rayset_by_marker_key,
            age_from_value_function=MarkerRaySet.age_seconds,
            query_timestamp=query_timestamp,
            maximum_age_seconds=self._minimum_marker_age_before_removal_seconds)
        changed |= modified
        self._alpha_poses_by_target_id, modified = self._clear_old_values_from_dict_of_lists(
            input_dict=self._alpha_poses_by_target_id,
            age_from_value_function=PoseData.age_seconds,
            query_timestamp=query_timestamp,
            maximum_age_seconds=self._parameters.POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS)
        changed |= modified
        self._target_extrapolation_poses_by_target_id, modified = self._clear_old_values_from_dict_of_lists(
            input_dict=self._target_extrapolation_poses_by_target_id,
            age_from_value_function=PoseData.age_seconds,
            query_timestamp=query_timestamp,
            maximum_age_seconds=self._parameters.POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS)
        changed |= modified
        self._target_depths_by_target_depth_key, modified = self._clear_old_values_from_dict_of_lists(
            input_dict=self._target_depths_by_target_depth_key,
            age_from_value_function=TargetDepth.age_seconds,
            query_timestamp=query_timestamp,
            maximum_age_seconds=self._parameters.POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS)
        changed |= modified
        return changed

    @staticmethod
    def _clear_old_values_from_dict(
        input_dict: dict[KeyType, ValueType],
        age_from_value_function: Callable[[ValueType, datetime.datetime], float],
        query_timestamp: datetime.datetime,
        maximum_age_seconds: float
    ) -> tuple[dict[KeyType, ValueType], bool]:  # modified_dictionary, changes_found
        changed: bool = False
        output_dict: dict[KeyType, ValueType] = dict()
        for input_key, input_value in input_dict.items():
            age_seconds: float = age_from_value_function(input_value, query_timestamp)
            if age_seconds <= maximum_age_seconds:
                output_dict[input_key] = input_value
            else:
                changed = True
        return output_dict, changed

    @staticmethod
    def _clear_old_values_from_dict_of_lists(
            input_dict: dict[KeyType, list[ValueType]],
            age_from_value_function: Callable[[ValueType, datetime.datetime], float],
            query_timestamp: datetime.datetime,
            maximum_age_seconds: float
    ) -> tuple[dict[KeyType, list[ValueType]], bool]:  # modified_dictionary, changes_found
        changed: bool = False
        output_dict: dict[KeyType, list[ValueType]] = dict()
        for input_key in input_dict.keys():
            output_poses_for_label: list[ValueType] = list()
            for pose in input_dict[input_key]:
                age_seconds: float = age_from_value_function(pose, query_timestamp)
                if age_seconds <= maximum_age_seconds:
                    output_poses_for_label.append(pose)
                else:
                    changed = True
            output_dict[input_key] = output_poses_for_label
        return output_dict, changed

    def _corresponding_point_list_in_target(
        self,
        target_id: uuid.UUID
    ) -> list[list[float]]:
        points = list()
        if target_id not in self._targets:
            raise RuntimeError(f"Could not find target {str(target_id)} in domain.")
        target: Target = self._targets[target_id]
        if isinstance(target, TargetMarker):
            half_width = target.marker_size / 2.0
            points += [
                [-half_width, half_width, 0.0],
                [half_width, half_width, 0.0],
                [half_width, -half_width, 0.0],
                [-half_width, -half_width, 0.0]]
        return points

    def _estimate_target_pose_from_ray_set(
        self,
        target: Target,
        ray_set: MarkerRaySet
    ) -> tuple[list[float], list[float]]:
        corners = numpy.array([ray_set.image_points], dtype="float32")
        detector_label: str = ray_set.detector_label
        camera_matrix: numpy.ndarray = numpy.array(
            self._intrinsics_by_detector_label[detector_label].get_matrix(), dtype="float32")
        distortion_coefficients: numpy.ndarray = numpy.array(
            self._intrinsics_by_detector_label[detector_label].get_distortion_coefficients(), dtype="float32")
        rotation_vector: numpy.ndarray
        translation_vector: numpy.ndarray
        if isinstance(target, TargetMarker):
            half_width = target.marker_size / 2.0
            reference_points: numpy.ndarray = numpy.array([
                [-half_width,  half_width, 0.0],
                [ half_width,  half_width, 0.0],
                [ half_width, -half_width, 0.0],
                [-half_width, -half_width, 0.0]],
                dtype="float32")
            reference_points = numpy.reshape(reference_points, newshape=(1, 4, 3))
            corners = numpy.reshape(corners, newshape=(1, 4, 2))
            _, rotation_vector, translation_vector = cv2.solvePnP(
                objectPoints=reference_points,
                imagePoints=corners,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients)
        else:
            raise NotImplementedError("Only targets that are boards or markers are supported.")

        rotation_vector = rotation_vector.flatten()
        translation_vector = translation_vector.flatten()
        object_to_camera_matrix = numpy.identity(4, dtype="float32")
        object_to_camera_matrix[0:3, 0:3] = Rotation.from_rotvec(rotation_vector).as_matrix()
        object_to_camera_matrix[0:3, 3] = translation_vector[0:3]
        object_to_detector_matrix = transformation_image_to_opengl(object_to_camera_matrix)
        detector_to_reference_matrix: Matrix4x4 = ray_set.detector_to_reference_matrix
        object_to_reference_matrix = numpy.matmul(
            detector_to_reference_matrix.as_numpy_array(), object_to_detector_matrix)
        position = list(object_to_reference_matrix[0:3, 3])
        quaternion = list(Rotation.from_matrix(object_to_reference_matrix[0:3, 0:3]).as_quat(canonical=True))
        return position, quaternion

    def _estimate_detector_pose_relative_to_reference(self):
        image_point_sets_by_image_key, image_point_set_keys_with_reference_visible = self._update()
        for image_point_sets_key, image_point_sets in image_point_sets_by_image_key.items():
            detector_label = image_point_sets_key.detector_label
            board_image_point_sets = [
                image_point_set for image_point_set in image_point_sets
                if image_point_set.marker_id in self._board_marker_ids
            ]

            # Create a dictionary to map marker_id to its index in _board_marker_ids
            marker_id_to_index = {marker_id: index for index, marker_id in enumerate(self._board_marker_ids)}

            # Sort the board_image_point_sets based on the order in _board_marker_ids
            board_image_point_sets.sort(key=lambda x: marker_id_to_index[x.marker_id])

            intrinsics: IntrinsicParameters = self._intrinsics_by_detector_label[detector_label]

            image_points = []
            detected_marker_positions = []
            for image_point_set in board_image_point_sets:
                image_points += image_point_set.points
                detected_marker_positions.append(
                    self._board_marker_positions[marker_id_to_index[image_point_set.marker_id]])

            if len(detected_marker_positions) == 0:
                continue  # Skip if no markers are detected

            half_width: float = self._board_marker_size / 2.0
            reference_points = []
            for position in detected_marker_positions:
                single_reference_points = numpy.array([
                    [position[0] + half_width, position[1] - half_width, 0.0],
                    [position[0] + half_width, position[1] + half_width, 0.0],
                    [position[0] - half_width, position[1] + half_width, 0.0],
                    [position[0] - half_width, position[1] - half_width, 0.0],
                ])
                reference_points.extend(single_reference_points)

            reference_points = numpy.array(reference_points, dtype="float32")
            reference_points = numpy.reshape(reference_points, newshape=(1, len(reference_points), 3))
            image_points = numpy.reshape(numpy.array(image_points, dtype="float32"), newshape=(1, len(image_points), 2))

            reference_found: bool
            rotation_vector: numpy.ndarray
            translation_vector: numpy.ndarray
            reference_found, rotation_vector, translation_vector = cv2.solvePnP(
                objectPoints=reference_points,
                imagePoints=image_points,
                cameraMatrix=numpy.asarray(intrinsics.get_matrix(), dtype="float32"),
                distCoeffs=numpy.asarray(intrinsics.get_distortion_coefficients(), dtype="float32"))
            if not reference_found:
                continue  # Camera does not see reference board

            rotation_vector = rotation_vector.flatten()
            translation_vector = translation_vector.flatten()
            reference_to_camera_matrix = numpy.identity(4, dtype="float32")
            reference_to_camera_matrix[0:3, 0:3] = Rotation.from_rotvec(rotation_vector).as_matrix()
            reference_to_camera_matrix[0:3, 3] = translation_vector
            reference_to_detector_matrix = transformation_image_to_opengl(reference_to_camera_matrix)
            detector_to_reference_opengl = numpy.linalg.inv(reference_to_detector_matrix)
            self._poses_by_detector_label[detector_label] = Matrix4x4.from_numpy_array(detector_to_reference_opengl)

    def _estimate_target_pose_relative_to_reference(self):
        image_point_sets_by_image_key, image_point_set_keys_with_reference_visible = self._update()
        valid_image_point_sets: list[MarkerCorners] = list()
        for image_point_sets_key in image_point_set_keys_with_reference_visible:
            image_point_sets = image_point_sets_by_image_key[image_point_sets_key]
            for image_point_set in image_point_sets:
                valid_image_point_sets.append(image_point_set)

        # Calculate rays
        # Only the most recent points per detector/marker pair are used,
        # so if we process the most recent first, we can detect and
        # discard the older point sets and avoid unnecessary processing
        valid_image_point_sets.sort(key=lambda x: x.timestamp, reverse=True)
        for image_point_set in valid_image_point_sets:
            image_timestamp = image_point_set.timestamp
            marker_key = MarkerKey(
                detector_label=image_point_set.detector_label,
                marker_id=image_point_set.marker_id)
            if marker_key in self._marker_rayset_by_marker_key:
                if self._marker_rayset_by_marker_key[marker_key].image_timestamp > image_timestamp:
                    continue  # A newer timestamp was found in this iteration. Skip the older one.
            for pose in self._detector_poses:
                if pose.target_id == image_point_set.detector_label:
                    detector_to_reference_matrix: Matrix4x4 = pose.object_to_reference_matrix
                    self._marker_rayset_by_marker_key[marker_key] = self._calculate_marker_ray_set(
                        image_point_set=image_point_set,
                        detector_to_reference_matrix=detector_to_reference_matrix)

        # Create a dictionary that maps marker ID's to a list of *recent* rays
        ray_sets_by_marker_id: dict[int, list[MarkerRaySet]] = dict()
        for marker_key, marker_ray_set in self._marker_rayset_by_marker_key.items():
            if (self._now_timestamp - marker_ray_set.image_timestamp).total_seconds() > \
                    self._parameters.POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS:
                continue
            marker_id = marker_key.marker_id
            if marker_id not in ray_sets_by_marker_id:
                ray_sets_by_marker_id[marker_id] = list()
            ray_sets_by_marker_id[marker_id].append(marker_ray_set)

        # Sort rays by the size of quadrilateral.
        # Larger marker size in image suggests more precision.
        # After a certain number of intersections,
        # there may be little point in processing additional (lower precision) ray sets.
        for marker_id, ray_set_list in ray_sets_by_marker_id.items():
            ray_set_list.sort(key=lambda x: convex_quadrilateral_area(x.image_points), reverse=True)
            ray_sets_by_marker_id[marker_id] = ray_set_list[0:self._parameters.MAXIMUM_RAY_COUNT_FOR_INTERSECTION]

        marker_count_by_marker_id: dict[int, int] = dict()
        for marker_id, ray_set_list in ray_sets_by_marker_id.items():
            marker_count_by_marker_id[marker_id] = len(ray_set_list)
        intersectable_marker_ids: list[int] = list()
        nonintersectable_marker_ids: list[int] = list()
        for marker_id, count in marker_count_by_marker_id.items():
            if count >= 2:
                intersectable_marker_ids.append(marker_id)
            else:
                nonintersectable_marker_ids.append(marker_id)

        # intersect rays to find the 3D points for each marker corner in reference coordinates
        corner_sets_reference_by_marker_id: dict[int, CornerSetReference] = dict()
        rejected_intersection_marker_ids: list[int] = list()
        for marker_id in intersectable_marker_ids:
            intersections_appear_valid: bool = True  # If something looks off, set this to False
            ray_set_list: list[MarkerRaySet] = ray_sets_by_marker_id[marker_id]
            corner_points_in_reference: list[list[float]] = list()
            for corner_index in range(0, 4):
                rays: list[Ray] = list()
                if len(ray_set_list) == 0:
                    intersections_appear_valid = False
                    print("Warning: intersectable_marker_ids corresponds to no ray set list")
                    break

                for ray_set in ray_set_list:
                    rays.append(Ray(
                        source_point=ray_set.ray_origin_reference,
                        direction=ray_set.ray_directions_reference[corner_index]))
                intersection_result = closest_intersection_between_n_lines(
                    rays=rays,
                    maximum_distance=self._parameters.INTERSECTION_MAXIMUM_DISTANCE)
                if intersection_result.centroids.shape[0] == 0:
                    intersections_appear_valid = False
                    break
                else:
                    corner_points_in_reference.append(intersection_result.centroid())
            if not intersections_appear_valid:
                rejected_intersection_marker_ids.append(marker_id)
                continue
            corner_sets_reference_by_marker_id[marker_id] = CornerSetReference(
                marker_id=marker_id,
                corners=corner_points_in_reference,
                ray_sets=ray_set_list)

        # We estimate the pose of each target based on the calculated intersections
        # and the rays projected from each detector
        for target_id, target in self._targets.items():
            marker_ids_in_target: list[int]
            if isinstance(target, TargetMarker):
                marker_ids_in_target = [target.marker_id]
            else:
                raise NotImplementedError("Only targets that are markers are supported.")

            marker_ids_with_intersections: list[int] = list()
            marker_ids_with_rays: list[int] = list()
            for marker_id in marker_ids_in_target:
                if marker_id in corner_sets_reference_by_marker_id:
                    marker_ids_with_intersections.append(marker_id)
                elif marker_id in ray_sets_by_marker_id:  # Don't include if we have (presumably precise) intersections
                    marker_ids_with_rays.append(marker_id)

            if len(marker_ids_with_intersections) <= 0 and len(marker_ids_with_rays) <= 0:
                continue  # No information on which to base a pose

            # Determine how many markers and how many detectors are involved
            marker_id_set: set[int] = set()
            one_detector_only: bool = True
            detector_set: set[str] = set()
            ray_sets: list[MarkerRaySet] = list()
            for marker_id in marker_ids_with_intersections:
                marker_id_set.add(corner_sets_reference_by_marker_id[marker_id].marker_id)
                ray_sets += corner_sets_reference_by_marker_id[marker_id].ray_sets
                one_detector_only = False
            for marker_id in marker_ids_with_rays:
                marker_id_set.add(marker_id)
                ray_sets += ray_sets_by_marker_id[marker_id]
            assert (len(marker_id_set) > 0)
            one_marker_only: bool = len(marker_id_set) == 1

            for ray_set in ray_sets:
                detector_set.add(ray_set.detector_label)
            one_detector_only &= (len(detector_set) == 1)

            # Try to find a solution for this matrix
            object_to_reference_matrix: numpy.array = numpy.identity(4, dtype="float32")

            if one_detector_only and one_marker_only:
                marker_id = marker_ids_with_rays[0]
                ray_set = ray_sets_by_marker_id[marker_id][0]
                position, orientation = self._estimate_target_pose_from_ray_set(
                    target=target,
                    ray_set=ray_set)
                object_to_reference_matrix[0:3, 3] = position
                object_to_reference_matrix[0:3, 0:3] = Rotation.from_quat(orientation).as_matrix()

            else:
                # Fill in the required variables for the customized iterative closest point
                initial_object_to_reference_estimated: bool = False
                initial_object_to_reference_matrix = numpy.identity(4, dtype="float32")
                object_known_points: list[list[float]] = list()
                reference_known_points: list[list[float]] = list()
                object_ray_points: list[list[float]] = list()
                reference_rays: list[Ray] = list()
                iterative_closest_point_parameters = IterativeClosestPointParameters(
                    termination_iteration_count=self._parameters.ITERATIVE_CLOSEST_POINT_TERMINATION_ITERATION_COUNT,
                    termination_delta_translation=self._parameters.ITERATIVE_CLOSEST_POINT_TERMINATION_TRANSLATION,
                    termination_delta_rotation_radians=self._parameters.ITERATIVE_CLOSEST_POINT_TERMINATION_ROTATION_RADIANS,
                    termination_mean_point_distance=self._parameters.ITERATIVE_CLOSEST_POINT_TERMINATION_MEAN_POINT_DISTANCE,
                    termination_rms_point_distance=self._parameters.ITERATIVE_CLOSEST_POINT_TERMINATION_RMS_POINT_DISTANCE)

                if len(marker_ids_with_intersections) >= 1:
                    reference_points_for_intersections: list[list[float]] = list()
                    for marker_id in marker_ids_with_intersections:
                        corner_set_reference = corner_sets_reference_by_marker_id[marker_id]
                        reference_points_for_intersections += corner_set_reference.corners
                    object_points_for_intersections = self._corresponding_point_list_in_target(target_id=target_id)
                    object_known_points += object_points_for_intersections
                    reference_known_points += reference_points_for_intersections
                    initial_object_to_reference_matrix = register_corresponding_points(
                        point_set_from=object_points_for_intersections,
                        point_set_to=reference_points_for_intersections)
                    initial_object_to_reference_estimated = True

                # pose estimation based on ArUco directly, used *only* for initial pose estimation
                estimated_positions: list[list[float]] = list()
                estimated_orientations: list[list[float]] = list()  # quaternions
                for marker_id in marker_ids_with_rays:
                    ray_set_list = ray_sets_by_marker_id[marker_id]
                    for ray_set in ray_set_list:
                        assert (len(ray_set.ray_directions_reference) == 4)
                        reference_rays_for_set: list[Ray] = list()
                        for corner_index in range(0, 4):
                            reference_rays_for_set.append(Ray(
                                source_point=ray_set.ray_origin_reference,
                                direction=ray_set.ray_directions_reference[corner_index]))
                        reference_rays += reference_rays_for_set
                        object_points_for_set = self._corresponding_point_list_in_target(target_id=target_id)
                        object_ray_points += object_points_for_set
                        if not initial_object_to_reference_estimated:
                            position, orientation = self._estimate_target_pose_from_ray_set(target, ray_set)
                            estimated_positions.append(position)
                            estimated_orientations.append(orientation)
                if not initial_object_to_reference_estimated:
                    mean_position = numpy.array([0.0, 0.0, 0.0])
                    for position in estimated_positions:
                        mean_position += position
                    mean_position /= len(estimated_positions)
                    initial_object_to_reference_matrix[0:3, 3] = mean_position
                    mean_orientation = average_quaternion(estimated_orientations)
                    initial_object_to_reference_matrix[0:3, 0:3] = Rotation.from_quat(mean_orientation).as_matrix()

                icp_output = iterative_closest_point_for_points_and_rays(
                    source_known_points=object_known_points,
                    target_known_points=reference_known_points,
                    source_ray_points=object_ray_points,
                    target_rays=reference_rays,
                    initial_transformation_matrix=initial_object_to_reference_matrix,
                    parameters=iterative_closest_point_parameters)
                object_to_reference_matrix = icp_output.source_to_target_matrix

            # Compute a depth from each detector,
            # find newest ray_set for each detector
            newest_ray_set_by_detector_label: dict[str, MarkerRaySet] = dict()
            for ray_set in ray_sets:
                detector_label = ray_set.detector_label
                if detector_label not in newest_ray_set_by_detector_label:
                    newest_ray_set_by_detector_label[detector_label] = ray_set
                elif ray_set.image_timestamp > newest_ray_set_by_detector_label[detector_label].image_timestamp:
                    newest_ray_set_by_detector_label[detector_label] = ray_set
            # Record depth
            for detector_label in newest_ray_set_by_detector_label:
                newest_ray_set = newest_ray_set_by_detector_label[detector_label]
                target_depth_key = TargetDepthKey(target_id=target_id, detector_label=detector_label)
                if target_depth_key not in self._target_depths_by_target_depth_key:
                    self._target_depths_by_target_depth_key[target_depth_key] = list()
                detector_to_reference_matrix: Matrix4x4 = newest_ray_set.detector_to_reference_matrix
                detector_position_reference: numpy.ndarray = detector_to_reference_matrix.as_numpy_array()[0:3, 3]
                object_position_reference: numpy.array = object_to_reference_matrix[0:3, 3]
                depth = numpy.linalg.norm(object_position_reference - detector_position_reference)
                target_depth = TargetDepth(
                    target_id=target_id,
                    detector_label=detector_label,
                    estimate_timestamp=newest_ray_set.image_timestamp,
                    depth=depth)
                self._target_depths_by_target_depth_key[target_depth_key].append(target_depth)
            # If only visible to one camera, use the depth to denoise
            if one_detector_only:
                detector_label = detector_set.pop()
                detector_to_reference_matrix: Matrix4x4 = \
                    newest_ray_set_by_detector_label[detector_label].detector_to_reference_matrix
                detector_position_reference = detector_to_reference_matrix.as_numpy_array()[0:3, 3]
                target_position_reference = object_to_reference_matrix[0:3, 3]
                depth_vector_reference = target_position_reference - detector_position_reference
                old_depth = numpy.linalg.norm(depth_vector_reference)
                target_depth_key = TargetDepthKey(target_id=target_id, detector_label=detector_label)
                new_depth = float(numpy.average(
                    [target_depth.depth for target_depth in
                     self._target_depths_by_target_depth_key[
                         target_depth_key]])) + self._parameters.POSE_SINGLE_CAMERA_DEPTH_CORRECTION
                depth_factor = new_depth / old_depth
                object_to_reference_matrix[0:3, 3] = detector_position_reference + depth_factor * depth_vector_reference

            pose = PoseData(
                target_id=target_id,
                object_to_reference_matrix=Matrix4x4.from_numpy_array(object_to_reference_matrix),
                ray_sets=ray_sets)

            if target_id not in self._target_extrapolation_poses_by_target_id:
                self._target_extrapolation_poses_by_target_id[target_id] = list()
            self._target_extrapolation_poses_by_target_id[target_id].append(pose)

            self._poses_by_target_id[target_id] = pose

    def _update(self):
        now_timestamp = datetime.datetime.now()
        self._now_timestamp = now_timestamp
        poses_need_update: bool = self._clear_old_values(now_timestamp)
        poses_need_update |= len(self._marker_corners_since_update) > 0
        if not poses_need_update:
            return

        self._poses_by_target_id.clear()

        image_point_sets_by_image_key: dict[ImagePointSetsKey, list[MarkerCorners]] = dict()
        for marker_corners in self._marker_corners_since_update:
            detector_label = marker_corners.detector_label
            image_point_sets_key = ImagePointSetsKey(detector_label, marker_corners.timestamp)
            if image_point_sets_key not in image_point_sets_by_image_key:
                image_point_sets_by_image_key[image_point_sets_key] = list()
            image_point_sets_by_image_key[image_point_sets_key].append(marker_corners)
        self._marker_corners_since_update.clear()

        image_point_set_keys_with_reference_visible: list[ImagePointSetsKey] = list()
        for image_point_sets_key, image_point_sets in image_point_sets_by_image_key.items():
            image_point_set_keys_with_reference_visible.append(image_point_sets_key)

        return image_point_sets_by_image_key, image_point_set_keys_with_reference_visible

