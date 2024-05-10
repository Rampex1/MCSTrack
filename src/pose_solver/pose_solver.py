from .exceptions import \
    PoseSolverException
from .structures import \
    MarkerCorners, \
    MarkerRaySet, \
    PoseData, \
    Ray, \
    Target, \
    TargetMarker
from .util import \
    average_quaternion, \
    average_vector, \
    convex_quadrilateral_area, \
    closest_intersection_between_n_lines, \
    IterativeClosestPointParameters, \
    iterative_closest_point_for_points_and_rays, \
    register_corresponding_points, \
    transformation_image_to_opengl, \
    vector_image_to_opengl
from src.common.structures import \
    IntrinsicParameters, \
    Matrix4x4, \
    Pose
import cv2
import cv2.aruco
import datetime
import numpy
from scipy.spatial.transform import Rotation
from typing import Callable, Final, Optional, TypeVar
import uuid


# TODO: These can be made into a parameter struct/file input
MAXIMUM_RAY_COUNT_FOR_INTERSECTION: Final[int] = 2
POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS: Final[float] = 0.1
POSE_SINGLE_CAMERA_EXTRAPOLATION_MINIMUM_SURFACE_NORMAL_ANGLE_DEGREES: Final[float] = 15.0
POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS: Final[float] = 1.0
POSE_SINGLE_CAMERA_EXTRAPOLATION_MAXIMUM_ORDER: Final[int] = 0
POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_ANGLE_DEGREES: Final[float] = 15.0
POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_DISTANCE: Final[float] = 15.0  # millimeters
POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS: Final[float] = 0.8
POSE_SINGLE_CAMERA_NEAREST_LIMIT_ANGLE_DEGREES: Final[float] = 15.0
POSE_SINGLE_CAMERA_NEAREST_LIMIT_DISTANCE: Final[float] = 15.0  # millimeters
POSE_SINGLE_CAMERA_REPROJECTION_ERROR_FACTOR_BETA_OVER_ALPHA: Final[float] = 1.0
POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS: Final[float] = 0.4
# TODO: Is this next one detector-specific?
POSE_SINGLE_CAMERA_DEPTH_CORRECTION: Final[float] = -7.5  # millimeters, observed tendency to overestimate depth
POSE_DETECTOR_DENOISE_LIMIT_AGE_SECONDS: Final[float] = 1.0
INTERSECTION_MAXIMUM_DISTANCE: Final[float] = 10  # millimeters
ITERATIVE_CLOSEST_POINT_TERMINATION_ITERATION_COUNT: Final[int] = 50
ITERATIVE_CLOSEST_POINT_TERMINATION_TRANSLATION: Final[float] = 0.005  # millimeters
ITERATIVE_CLOSEST_POINT_TERMINATION_ROTATION_RADIANS: Final[float] = 0.0005
ITERATIVE_CLOSEST_POINT_TERMINATION_MEAN_POINT_DISTANCE: Final[float] = 0.1  # millimeters
ITERATIVE_CLOSEST_POINT_TERMINATION_RMS_POINT_DISTANCE: Final[float] = 0.1  # millimeters

DENOISE_OUTLIER_DISTANCE_MILLIMETERS: Final[float] = 10.0
DENOISE_OUTLIER_ANGLE_DEGREES: Final[float] = 5.0
DENOISE_STORAGE_SIZE: Final[int] = 10
DENOISE_FILTER_SIZE: Final[int] = 7
DENOISE_REQUIRED_STARTING_STREAK: Final[int] = 3

ARUCO_MARKER_DICTIONARY_ENUM: Final[int] = cv2.aruco.DICT_4X4_100
ARUCO_POSE_ESTIMATOR_METHOD: Final[int] = cv2.SOLVEPNP_ITERATIVE  # TODO: use this value in solvePnP calls
# SOLVEPNP_ITERATIVE works okay but is susceptible to optical illusions (flipping)
# SOLVEPNP_P3P appears to return nan's on rare occasion
# SOLVEPNP_SQPNP appears to return nan's on rare occasion
# SOLVEPNP_IPPE_SQUARE does not seem to work very well at all, translation is much smaller than expected

EPSILON: Final[float] = 0.0001


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


class PoseExtrapolationQuality:
    plausible: bool
    score: float

    @staticmethod
    def quality_of_pose_extrapolation(
        expected_translation: numpy.array,
        expected_rotation_quaternion: numpy.array,
        sample_translation: numpy.array,
        sample_rotation_quaternion: numpy.array,
        maximum_relative_translation_magnitude: float,
        maximum_relative_rotation_magnitude_degrees: float
    ):  # -> PoseExtrapolationQuality
        expected_rotation = Rotation.from_quat(expected_rotation_quaternion)
        expected_inv_rotation = Rotation.inv(expected_rotation)
        relative_translation = sample_translation - expected_translation
        relative_translation_magnitude = numpy.linalg.norm(relative_translation)
        relative_rotation = expected_inv_rotation * Rotation.from_quat(sample_rotation_quaternion)
        relative_rotation_vector_degrees = relative_rotation.as_rotvec() * 180.0 / numpy.pi
        relative_rotation_magnitude_degrees = numpy.linalg.norm(relative_rotation_vector_degrees)
        translation_score = 1.0 - \
            (relative_translation_magnitude / maximum_relative_translation_magnitude)
        rotation_score = 1.0 - \
            (relative_rotation_magnitude_degrees / maximum_relative_rotation_magnitude_degrees)
        quality = PoseExtrapolationQuality()
        quality.plausible = (translation_score >= 0.0 and rotation_score >= 0.0)
        quality.score = min(translation_score, rotation_score)
        return quality


class PoseSolver:
    """
    Class containing the actual "solver" logic, kept separate from the API.
    """
    _intrinsics_by_detector_label: dict[str, IntrinsicParameters]
    _targets: dict[uuid.UUID, Target]
    _reference_target: Target

    _marker_corners_since_update: list[MarkerCorners]

    # Not all images arrive at the same time.
    # This structure provides a means for storing ray information between frames,
    # so that the frames do not need to arrive at the same time to resolve the pose.
    _marker_rayset_by_marker_key: dict[MarkerKey, MarkerRaySet]

    # These variables store histories of poses, generally for denoising or resolving pose ambiguities
    _alpha_poses_by_target_id: dict[uuid.UUID, list[PoseData]]  # Found from a single camera
    _target_extrapolation_poses_by_target_id: dict[uuid.UUID, list[PoseData]]  # Used for extrapolation, single camera
    _poses_by_target_id: dict[uuid.UUID, PoseData]
    _poses_by_detector_label: dict[str, Matrix4x4]
    _target_depths_by_target_depth_key: dict[TargetDepthKey, list[TargetDepth]]

    _minimum_marker_age_before_removal_seconds: float

    def __init__(
        self
    ):
        # TODO: Endpoint to handle detector sending data. Each time a detector adds points etc,
        #       We must check the dict to see if it has been registered or not.

        # TODO: Reference can be a board, temporarily hardcoded to ArUco marker with ID 0.

        # TODO: A means of adding target markers (?)

        self._intrinsics_by_detector_label = dict()
        self._targets = dict()
        self._reference_target = TargetMarker(
            marker_id=0,
            marker_size=10)
        self._marker_corners_since_update = list()
        self._marker_rayset_by_marker_key = dict()
        self._alpha_poses_by_target_id = dict()
        self._target_extrapolation_poses_by_target_id = dict()
        self._poses_by_target_id = dict()
        self._poses_by_detector_label = dict()
        self._target_depths_by_target_depth_key = dict()

        self._minimum_marker_age_before_removal_seconds = max([
            POSE_DETECTOR_DENOISE_LIMIT_AGE_SECONDS,
            POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS,
            POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS,
            POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS,
            POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS])

    def add_marker_corners(
        self,
        detected_corners: list[MarkerCorners]
    ) -> None:
        self._marker_corners_since_update += detected_corners

    def add_target_marker(
        self,
        marker_id: int,
        marker_diameter: float
    ) -> str:
        if isinstance(self._reference_target, TargetMarker) and self._reference_target.marker_id == marker_id:
            raise PoseSolverException(message=f"Marker with id {marker_id} is already the reference.")
        for target_id, target in self._targets.items():
            if isinstance(target, TargetMarker) and marker_id == target.marker_id:
                raise PoseSolverException(message=f"Marker with id {marker_id} is already being tracked.")
        target: Target = TargetMarker(
            marker_id=marker_id,
            marker_size=marker_diameter)
        target_id: uuid.UUID = uuid.uuid4()
        self._targets[target_id] = target
        return str(target_id)

    def target_marker_exists(
        self,
        marker_id: int,
        marker_diameter: float,
    ) -> bool:
        """
        Returns True if target_marker has already been detected (David)
        Note: Later on, can be combined with add_target_marker for simplified code
        """
        try:
            self.add_target_marker(marker_id=marker_id, marker_diameter=marker_diameter)
            print(f"Added {marker_id}")
            return True
        except PoseSolverException as e:
            print(f"{marker_id} already exists")
            return False

    def get_poses(
        self
    ) -> tuple[list[Pose], list[Pose]]:
        """
        Returns detector_poses, target_poses
        """
        detector_poses: list[Pose] = [
            Pose(
                target_id=detector_label,
                object_to_reference_matrix=pose,
                solver_timestamp_utc_iso8601=str(datetime.datetime.utcnow().isoformat()))  # TODO: real value
            for detector_label, pose in self._poses_by_detector_label.items()]
        target_poses: list[Pose] = [
            Pose(
                target_id=str(target_id),
                object_to_reference_matrix=pose.object_to_reference_matrix,
                solver_timestamp_utc_iso8601=str(pose.newest_timestamp().isoformat()))  # TODO: real value
            for target_id, pose in self._poses_by_target_id.items()]
        return detector_poses, target_poses

    def set_intrinsic_parameters(
        self,
        detector_label: str,
        intrinsic_parameters: IntrinsicParameters
    ) -> None:
        self._intrinsics_by_detector_label[detector_label] = intrinsic_parameters

    def set_reference_target(
        self,
        target: Target
    ) -> None:
        if not isinstance(target, TargetMarker):
            raise NotImplementedError("Only targets that are of type TargetMarker are currently supported.")
        for tracked_target_id, tracked_target in self._targets.items():
            if isinstance(tracked_target, TargetMarker) and target.marker_id == tracked_target.marker_id:
                # TODO: Notify that this tracked target is now the reference
                self._targets.pop(tracked_target_id)
                break
        self._reference_target = target

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

    def _calculate_reprojection_error_for_pose(
        self,
        ray_set: MarkerRaySet,
        object_points_target: list[list[float]],
        object_to_reference_translation: list[float],
        object_to_reference_rotation_quaternion: list[float]
    ) -> float:
        object_to_reference_matrix = numpy.identity(4, dtype="float32")
        object_to_reference_matrix[0:3, 0:3] = Rotation.from_quat(object_to_reference_rotation_quaternion).as_matrix()
        object_to_reference_matrix[0:3, 3] = object_to_reference_translation
        object_points_reference = numpy.empty((len(object_points_target), 3), dtype="float32")
        for object_point_index, object_point_target in enumerate(object_points_target):
            object_point_reference = numpy.matmul(
                object_to_reference_matrix,
                [object_point_target[0], object_point_target[1], object_point_target[2], 1.0])
            object_points_reference[object_point_index, 0:3] = object_point_reference[0:3]
        detector_label: str = ray_set.detector_label
        reference_to_detector_matrix: Matrix4x4 = ray_set.detector_to_reference_matrix
        reference_to_detector: numpy.ndarray = reference_to_detector_matrix.as_numpy_array()
        reference_to_detector_rotation_vector = \
            Rotation.as_rotvec(Rotation.from_matrix(reference_to_detector[0:3, 0:3]))
        reference_to_detector_translation = reference_to_detector[0:3, 3]
        camera_matrix = numpy.array(self._intrinsics_by_detector_label[detector_label].get_matrix(), dtype="float32")
        camera_distortion_coefficients = \
            numpy.array(self._intrinsics_by_detector_label[detector_label].get_distortion_coefficients(),
                        dtype="float32")
        project_points_result = cv2.projectPoints(
            objectPoints=object_points_reference,
            rvec=reference_to_detector_rotation_vector,
            tvec=reference_to_detector_translation,
            cameraMatrix=camera_matrix,
            distCoeffs=camera_distortion_coefficients)
        projected_points = project_points_result[0]
        sum_reprojection_errors_squared: float = 0.0
        for point_index, image_point in enumerate(ray_set.image_points):
            reprojection_error_for_point = \
                numpy.linalg.norm(projected_points[point_index, 0, 0:2] - image_point)
            sum_reprojection_errors_squared += reprojection_error_for_point ** 2
        mean_reprojection_errors_squared: float = sum_reprojection_errors_squared / len(object_points_target)
        rms_reprojection_error = numpy.sqrt(mean_reprojection_errors_squared)
        return rms_reprojection_error

    # noinspection DuplicatedCode
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
            maximum_age_seconds=POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS)
        changed |= modified
        self._target_extrapolation_poses_by_target_id, modified = self._clear_old_values_from_dict_of_lists(
            input_dict=self._target_extrapolation_poses_by_target_id,
            age_from_value_function=PoseData.age_seconds,
            query_timestamp=query_timestamp,
            maximum_age_seconds=POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS)
        changed |= modified
        self._target_depths_by_target_depth_key, modified = self._clear_old_values_from_dict_of_lists(
            input_dict=self._target_depths_by_target_depth_key,
            age_from_value_function=TargetDepth.age_seconds,
            query_timestamp=query_timestamp,
            maximum_age_seconds=POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS)
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

    @staticmethod
    def _denoise_is_pose_pair_outlier(
        pose_a: PoseData,
        pose_b: PoseData
    ) -> bool:

        position_a = pose_a.object_to_reference_matrix[0:3, 3]
        position_b = pose_b.object_to_reference_matrix[0:3, 3]
        distance_millimeters = numpy.linalg.norm(position_a - position_b)
        if distance_millimeters > DENOISE_OUTLIER_DISTANCE_MILLIMETERS:
            return True

        orientation_a = pose_a.object_to_reference_matrix[0:3, 0:3]
        orientation_b = pose_b.object_to_reference_matrix[0:3, 0:3]
        rotation_a_to_b = numpy.matmul(orientation_a, numpy.linalg.inv(orientation_b))
        angle_degrees = numpy.linalg.norm(Rotation.from_matrix(rotation_a_to_b).as_rotvec())
        if angle_degrees > DENOISE_OUTLIER_DISTANCE_MILLIMETERS:
            return True

        return False

    # Take an average of recent poses, with detection and removal of outliers
    # TODO: Currently not used - but it's probably a good feature to have
    def _denoise_detector_to_reference_pose(
        self,
        object_label: str,
        raw_poses: list[PoseData]  # In order, oldest to newest
    ) -> PoseData:
        most_recent_pose = raw_poses[-1]
        max_storage_size: int = DENOISE_STORAGE_SIZE
        filter_size: int = DENOISE_FILTER_SIZE
        if filter_size <= 1 or max_storage_size <= 1:
            return most_recent_pose  # trivial case

        # find a consistent range of recent indices
        poses: list[PoseData] = list(raw_poses)
        poses.reverse()  # now they are sorted so that the first element is most recent
        required_starting_streak: int = DENOISE_REQUIRED_STARTING_STREAK
        starting_index: int = -1  # not yet known, we want to find this
        if required_starting_streak <= 1:
            starting_index = 0  # trivial case
        else:
            current_inlier_streak: int = 1  # comparison always involves a streak of size 1 or more
            for pose_index in range(1, len(raw_poses)):
                if self._denoise_is_pose_pair_outlier(poses[pose_index - 1], poses[pose_index]):
                    current_inlier_streak = 1
                    continue
                current_inlier_streak += 1
                if current_inlier_streak >= required_starting_streak:
                    starting_index = pose_index - required_starting_streak + 1
                    break

        if starting_index < 0:
            if len(poses) >= max_storage_size:  # There appear to be enough poses, so data seems to be inconsistent
                print(
                    "Warning: Can't find consistent pose streak. "
                    "Will use most recent raw pose for object " + object_label + ".")
            return most_recent_pose

        poses_to_average: list[PoseData] = [poses[starting_index]]
        for pose_index in range(starting_index + 1, len(poses)):
            if (not self._denoise_is_pose_pair_outlier(poses[pose_index - 1], poses[pose_index])) or \
               (not self._denoise_is_pose_pair_outlier(poses[starting_index], poses[pose_index])):
                poses_to_average.append(poses[pose_index])
                if len(poses_to_average) > filter_size:
                    break

        translations = [list(pose.object_to_reference_matrix[0:3, 3])
                        for pose in poses_to_average]
        orientations = [list(Rotation.from_matrix(pose.object_to_reference_matrix[0:3, 0:3]).as_quat(canonical=True))
                        for pose in poses_to_average]
        filtered_translation = average_vector(translations)
        filtered_orientation = average_quaternion(orientations)
        filtered_object_to_reference_matrix = numpy.identity(4, dtype="float32")
        filtered_object_to_reference_matrix[0:3, 0:3] = Rotation.from_quat(filtered_orientation).as_matrix()
        filtered_object_to_reference_matrix[0:3, 3] = filtered_translation
        return PoseData(
            object_label=object_label,
            object_to_reference_matrix=filtered_object_to_reference_matrix,
            ray_sets=most_recent_pose.ray_sets)

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

    # Specialized version of _estimate_target_pose_from_ray_set for handling the pose ambiguity problem
    # noinspection PyUnreachableCode
    def _estimate_target_pose_from_ray_set_and_single_marker_id(
        self,
        ray_set: MarkerRaySet,
        target_id: uuid.UUID,
        target: Target
    ) -> tuple[list[float], list[float]]:
        assert (len(ray_set.ray_directions_reference) == 4)

        alpha_translation_reference, alpha_rotation_quaternion = self._estimate_target_pose_from_ray_set(
            ray_set=ray_set,
            target=target)

        # We don't know if the pose above is in the correct sense of the optical illusion problem.
        # We calculate the "other" version of the pose, then determine which is more likely to be correct.
        # Let "alpha" indicate the position obtained above, and "beta" indicate the other version
        alpha_pose_nparray: numpy.ndarray = numpy.identity(4, dtype="float32")
        alpha_pose_nparray[0:3, 0:3] = Rotation.from_quat(alpha_rotation_quaternion).as_matrix()
        alpha_pose_nparray[0:3, 3] = alpha_translation_reference
        alpha_pose_matrix: Matrix4x4 = Matrix4x4.from_numpy_array(alpha_pose_nparray)
        alpha_pose = PoseData(
            target_id=target_id,
            object_to_reference_matrix=alpha_pose_matrix,
            ray_sets=[ray_set])
        if target_id not in self._alpha_poses_by_target_id:
            self._alpha_poses_by_target_id[target_id] = list()
        self._alpha_poses_by_target_id[target_id].append(alpha_pose)

        # We need the original object points
        # First make sure they form a square (with reasonable tolerance)
        # because if not, then the following math becomes suspect...
        object_points = self._corresponding_point_list_in_target(target_id=target_id)
        if len(object_points) != 4:
            raise RuntimeError("Input marker points is of incorrect length, expected 4 got " + str(len(object_points)))
        point_top_left = numpy.array(object_points[0])
        point_top_right = numpy.array(object_points[1])
        point_bottom_right = numpy.array(object_points[2])
        point_bottom_left = numpy.array(object_points[3])
        marker_side = numpy.linalg.norm(point_top_right - point_top_left)
        if marker_side <= EPSILON:
            raise RuntimeError("Input marker points must define a square of side length larger than zero.")
        marker_diag = numpy.sqrt(numpy.square(marker_side) * 2)
        if abs(numpy.linalg.norm(point_bottom_right - point_top_right) - marker_side) > EPSILON or \
           abs(numpy.linalg.norm(point_bottom_left - point_bottom_right) - marker_side) > EPSILON or \
           abs(numpy.linalg.norm(point_top_left - point_bottom_left) - marker_side) > EPSILON or \
           abs(numpy.linalg.norm(point_top_left - point_bottom_right) - marker_diag) > EPSILON or \
           abs(numpy.linalg.norm(point_top_right - point_bottom_left) - marker_diag) > EPSILON:
            raise RuntimeError("Input marker points do not form a square.")

        # Calculate the rotation that creates the beta version of the pose
        axis_x_marker = \
            numpy.asarray(object_points[3], dtype="float32") - numpy.asarray(object_points[2], dtype="float32")
        axis_y_marker = \
            numpy.asarray(object_points[3], dtype="float32") - numpy.asarray(object_points[0], dtype="float32")
        axis_z_marker = numpy.cross(axis_x_marker, axis_y_marker)  # pointed out from marker
        rotation_alpha_to_reference_matrix = Rotation.from_quat(alpha_rotation_quaternion).as_matrix()
        alpha_axis_z_reference = numpy.matmul(rotation_alpha_to_reference_matrix, axis_z_marker)
        alpha_normal_reference = alpha_axis_z_reference / numpy.linalg.norm(alpha_axis_z_reference)
        marker_centroid_marker = numpy.array(average_vector(object_points), dtype="float32")
        marker_centroid_reference = \
            numpy.matmul(rotation_alpha_to_reference_matrix, marker_centroid_marker) + alpha_translation_reference
        ray_origin_reference = numpy.array(ray_set.ray_origin_reference, dtype="float32")
        marker_to_detector_reference = ray_origin_reference - marker_centroid_reference
        marker_to_ray_unit_reference = marker_to_detector_reference / numpy.linalg.norm(marker_to_detector_reference)
        rotation_alpha_to_reference_vector = numpy.cross(alpha_normal_reference, marker_to_ray_unit_reference)
        rotation_alpha_to_reference_magnitude_degrees = \
            numpy.linalg.norm(rotation_alpha_to_reference_vector) * 180.0 / numpy.pi
        if rotation_alpha_to_reference_magnitude_degrees <= EPSILON:
            # In the rare event where the view is perpendicular,
            # there is little point in computing beta since it will be almost identical to alpha
            return alpha_translation_reference, alpha_rotation_quaternion

        # The beta pose is an additional rotation so that the marker appears the same in the image
        # This seems like a bit of a simplification, and there may be a more precise way to calculate it
        rotation_reference_to_beta = rotation_alpha_to_reference_vector

        rotation_alpha_to_beta_vector = rotation_alpha_to_reference_vector + rotation_reference_to_beta
        rotation_alpha_to_beta = Rotation.from_rotvec(rotation_alpha_to_beta_vector)
        beta_rotation_quaternion = \
            (rotation_alpha_to_beta * Rotation.from_quat(alpha_rotation_quaternion)).as_quat(canonical=True)

        # Rotation about marker_centroid_reference, since rotation axis passes through it
        rotation_alpha_to_beta_matrix = rotation_alpha_to_beta.as_matrix()
        reference_to_alpha_translation_vector = \
            numpy.array(alpha_translation_reference, dtype="float32") - marker_centroid_reference
        reference_to_beta_translation_vector = numpy.matmul(
            rotation_alpha_to_beta_matrix, reference_to_alpha_translation_vector)
        beta_translation_reference = marker_centroid_reference + reference_to_beta_translation_vector

        # noinspection PyTypeChecker
        beta_rotation_quaternion = list(beta_rotation_quaternion.tolist())
        # noinspection PyTypeChecker
        beta_translation_reference = list(beta_translation_reference.tolist())

        alpha_pose_quality: Optional[PoseExtrapolationQuality] = None
        beta_pose_quality: Optional[PoseExtrapolationQuality] = None
        if rotation_alpha_to_reference_magnitude_degrees >= \
           POSE_SINGLE_CAMERA_EXTRAPOLATION_MINIMUM_SURFACE_NORMAL_ANGLE_DEGREES:
            # Use velocity and angular velocity over recent poses to extrapolate pose,
            # determine which of alpha and beta are closest (within threshold)
            recent_poses: list[PoseData] = list()
            target_label: str = str(target_id)
            if target_label in self._target_extrapolation_poses_by_target_id:
                for pose in self._target_extrapolation_poses_by_target_id[target.target_label]:
                    if (ray_set.image_timestamp - pose.oldest_timestamp()).total_seconds() <= \
                       POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS:
                        recent_poses.append(pose)
            if len(recent_poses) >= 4:  # Size of quaternion, minimum for solving linear equation
                oldest_pose_timestamp = sorted([pose.oldest_timestamp() for pose in recent_poses])[0]
                ts = numpy.empty((len(recent_poses), 1), dtype="float32")
                bs_quaternion = numpy.empty((len(recent_poses), 4), dtype="float32")
                bs_translation = numpy.empty((len(recent_poses), 3), dtype="float32")
                for pose_index, pose in enumerate(recent_poses):
                    ts[pose_index, 0] = (pose.oldest_timestamp() - oldest_pose_timestamp).total_seconds()
                    pose_quaternion = (Rotation
                                       .from_matrix(pose.object_to_reference_matrix[0:3, 0:3])
                                       .as_quat(canonical=True))
                    bs_quaternion[pose_index, 0:4] = pose_quaternion
                    bs_translation[pose_index, 0:3] = pose.object_to_reference_matrix[0:3, 3]
                maximum_order: int = 2
                a_matrix = numpy.ones((len(recent_poses), (maximum_order+1)), dtype="float32")
                for pose_index in range(0, len(recent_poses)):
                    for order in range(1, maximum_order+1):
                        a_matrix[pose_index, order] = ts[pose_index] ** order
                coefficients_quaternion = numpy.linalg.lstsq(a_matrix, bs_quaternion, rcond=None)[0]
                coefficients_translation = numpy.linalg.lstsq(a_matrix, bs_translation, rcond=None)[0]
                # Note in coefficients: order = 0..maximum_order
                extrapolation_t = (ray_set.image_timestamp - oldest_pose_timestamp).total_seconds()
                extrapolation_a = numpy.ones((1, (maximum_order+1)), dtype="float32")
                for order in range(1, (maximum_order+1)):
                    extrapolation_a[0, order] = extrapolation_t ** order
                extrapolated_rotation_quaternion = numpy.matmul(extrapolation_a, coefficients_quaternion)
                extrapolated_rotation_quaternion = \
                    extrapolated_rotation_quaternion / numpy.linalg.norm(extrapolated_rotation_quaternion)
                extrapolated_translation_reference = numpy.matmul(extrapolation_a, coefficients_translation)
                alpha_pose_quality = PoseExtrapolationQuality.quality_of_pose_extrapolation(
                    expected_translation=extrapolated_translation_reference,
                    expected_rotation_quaternion=extrapolated_rotation_quaternion,
                    sample_translation=numpy.array(alpha_translation_reference, dtype="float32"),
                    sample_rotation_quaternion=numpy.array(alpha_rotation_quaternion, dtype="float32"),
                    maximum_relative_translation_magnitude=POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_DISTANCE,
                    maximum_relative_rotation_magnitude_degrees=POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_ANGLE_DEGREES)
                beta_pose_quality = PoseExtrapolationQuality.quality_of_pose_extrapolation(
                    expected_translation=extrapolated_translation_reference,
                    expected_rotation_quaternion=extrapolated_rotation_quaternion,
                    sample_translation=numpy.array(beta_translation_reference, dtype="float32"),
                    sample_rotation_quaternion=numpy.array(beta_rotation_quaternion, dtype="float32"),
                    maximum_relative_translation_magnitude=POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_DISTANCE,
                    maximum_relative_rotation_magnitude_degrees=POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_ANGLE_DEGREES)
        else:
            if isinstance(target, TargetMarker):
                target_label = str(target.marker_id)
            else:
                raise NotImplementedError("Only targets that are markers are supported.")
            if len(self._alpha_poses_by_target_id[target_label]) > 4:
                # Look at the alpha poses over some recent duration, and then choose
                # alpha or beta that is (on average) closest to them,
                # optionally weighing more heavily recent measurements.
                alpha_translations_reference: list[list[float]] = list()
                alpha_rotations_quaternion: list[list[float]] = list()
                for alpha_pose in self._alpha_poses_by_target_id[target.target_label]:
                    if (ray_set.image_timestamp - alpha_pose.oldest_timestamp()).total_seconds() <= \
                       POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS:
                        alpha_translations_reference.append(list(
                            alpha_pose.object_to_reference_matrix[0:3, 3].tolist()))
                        # noinspection PyTypeChecker
                        alpha_rotations_quaternion.append(list(
                            Rotation
                            .from_matrix(alpha_pose.object_to_reference_matrix[0:3, 0:3])
                            .as_quat(canonical=True)
                            .tolist()))
                average_alpha_translation_reference = average_vector(alpha_translations_reference)
                average_alpha_rotation_quaternion = average_quaternion(alpha_rotations_quaternion)
                alpha_pose_quality = PoseExtrapolationQuality.quality_of_pose_extrapolation(
                    expected_translation=numpy.array(average_alpha_translation_reference, dtype="float32"),
                    expected_rotation_quaternion=numpy.array(average_alpha_rotation_quaternion, dtype="float32"),
                    sample_translation=numpy.array(alpha_translation_reference, dtype="float32"),
                    sample_rotation_quaternion=numpy.array(alpha_rotation_quaternion, dtype="float32"),
                    maximum_relative_translation_magnitude=POSE_SINGLE_CAMERA_NEAREST_LIMIT_DISTANCE,
                    maximum_relative_rotation_magnitude_degrees=POSE_SINGLE_CAMERA_NEAREST_LIMIT_ANGLE_DEGREES)
                beta_pose_quality = PoseExtrapolationQuality.quality_of_pose_extrapolation(
                    expected_translation=numpy.array(average_alpha_translation_reference, dtype="float32"),
                    expected_rotation_quaternion=numpy.array(average_alpha_rotation_quaternion, dtype="float32"),
                    sample_translation=numpy.array(beta_translation_reference, dtype="float32"),
                    sample_rotation_quaternion=numpy.array(beta_rotation_quaternion, dtype="float32"),
                    maximum_relative_translation_magnitude=POSE_SINGLE_CAMERA_NEAREST_LIMIT_DISTANCE,
                    maximum_relative_rotation_magnitude_degrees=POSE_SINGLE_CAMERA_NEAREST_LIMIT_ANGLE_DEGREES)

        if alpha_pose_quality is not None and beta_pose_quality is not None:
            if alpha_pose_quality.plausible and beta_pose_quality.plausible:
                if alpha_pose_quality.score >= beta_pose_quality.score:
                    return alpha_translation_reference, alpha_rotation_quaternion
                else:
                    return beta_translation_reference, beta_rotation_quaternion
            elif alpha_pose_quality.plausible:
                return alpha_translation_reference, alpha_rotation_quaternion
            elif beta_pose_quality.plausible:
                return beta_translation_reference, beta_rotation_quaternion

        alpha_reprojection_error = self._calculate_reprojection_error_for_pose(
            ray_set=ray_set,
            object_points_target=object_points,
            object_to_reference_translation=alpha_translation_reference,
            object_to_reference_rotation_quaternion=alpha_rotation_quaternion)
        beta_reprojection_error = self._calculate_reprojection_error_for_pose(
            ray_set=ray_set,
            object_points_target=object_points,
            object_to_reference_translation=beta_translation_reference,
            object_to_reference_rotation_quaternion=beta_rotation_quaternion)
        if beta_reprojection_error <= \
           POSE_SINGLE_CAMERA_REPROJECTION_ERROR_FACTOR_BETA_OVER_ALPHA * alpha_reprojection_error:
            return beta_translation_reference, beta_rotation_quaternion

        return alpha_translation_reference, alpha_rotation_quaternion

    def update(self):
        now_timestamp = datetime.datetime.now()
        poses_need_update: bool = self._clear_old_values(now_timestamp)
        poses_need_update |= len(self._marker_corners_since_update) > 0
        if not poses_need_update:
            return

        self._poses_by_detector_label.clear()
        self._poses_by_target_id.clear()

        image_point_sets_by_image_key: dict[ImagePointSetsKey, list[MarkerCorners]] = dict()
        for marker_corners in self._marker_corners_since_update:
            detector_label = marker_corners.detector_label
            image_point_sets_key = ImagePointSetsKey(detector_label, marker_corners.timestamp)
            if image_point_sets_key not in image_point_sets_by_image_key:
                image_point_sets_by_image_key[image_point_sets_key] = list()
            image_point_sets_by_image_key[image_point_sets_key].append(marker_corners)

        self._marker_corners_since_update.clear()

        # TODO: Remove this condition with the addition of board
        if not isinstance(self._reference_target, TargetMarker):
            return

        # estimate the detector pose relative to the reference target
        # TODO: Rather than calculate for all point sets *then* see which markers can be found,
        #       Find the most recent ImagePointSet for each marker,
        #       then compute reference pose *only* as needed
        image_point_set_keys_with_reference_visible: list[ImagePointSetsKey] = list()
        for image_point_sets_key, image_point_sets in image_point_sets_by_image_key.items():
            detector_label = image_point_sets_key.detector_label
            image_point_set_reference: MarkerCorners | None = None
            for image_point_set in image_point_sets:
                if image_point_set.marker_id == self._reference_target.marker_id:
                    image_point_set_reference = image_point_set
                    break
            if image_point_set_reference is None:
                continue  # Reference not visible
            intrinsics: IntrinsicParameters = self._intrinsics_by_detector_label[detector_label]
            half_width: float = self._reference_target.marker_size / 2.0
            reference_points: numpy.ndarray = numpy.array([
                [-half_width,  half_width, 0.0],
                [ half_width,  half_width, 0.0],
                [ half_width, -half_width, 0.0],
                [-half_width, -half_width, 0.0]],
                dtype="float32")
            reference_points = numpy.reshape(reference_points, newshape=(1, 4, 3))
            image_points: numpy.ndarray = numpy.array([image_point_set_reference.points], dtype="float32")
            image_points = numpy.reshape(image_points, newshape=(1, 4, 2))
            reference_found: bool
            rotation_vector: numpy.ndarray
            translation_vector: numpy.ndarray
            reference_found, rotation_vector, translation_vector = cv2.solvePnP(
                objectPoints=reference_points,
                imagePoints=image_points,
                cameraMatrix=numpy.asarray(intrinsics.get_matrix(), dtype="float32"),
                distCoeffs=numpy.asarray(intrinsics.get_distortion_coefficients(), dtype="float32"))
            if not reference_found:
                continue  # Camera does not see reference target
            rotation_vector = rotation_vector.flatten()
            translation_vector = translation_vector.flatten()
            reference_to_camera_matrix = numpy.identity(4, dtype="float32")
            reference_to_camera_matrix[0:3, 0:3] = Rotation.from_rotvec(rotation_vector).as_matrix()
            reference_to_camera_matrix[0:3, 3] = translation_vector
            reference_to_detector_matrix = transformation_image_to_opengl(reference_to_camera_matrix)
            detector_to_reference_opengl = numpy.linalg.inv(reference_to_detector_matrix)
            self._poses_by_detector_label[detector_label] = Matrix4x4.from_numpy_array(detector_to_reference_opengl)
            image_point_set_keys_with_reference_visible.append(image_point_sets_key)

        # Code beyond this point is for tracking targets other than the reference.
        # But since the targets are tracked relative to the reference,
        # the reference must be visible.
        # In addition, code beyond this is for tracking targets other than the reference.
        # Therefore no need to process image points/rays corresponding
        # to marker id's in the reference target.
        valid_image_point_sets: list[MarkerCorners] = list()
        for image_point_sets_key in image_point_set_keys_with_reference_visible:
            image_point_sets = image_point_sets_by_image_key[image_point_sets_key]
            for image_point_set in image_point_sets:
                if image_point_set.marker_id != self._reference_target.marker_id:
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
            detector_to_reference_matrix: Matrix4x4 = self._poses_by_detector_label[image_point_set.detector_label]
            self._marker_rayset_by_marker_key[marker_key] = self._calculate_marker_ray_set(
                image_point_set=image_point_set,
                detector_to_reference_matrix=detector_to_reference_matrix)

        # Create a dictionary that maps marker ID's to a list of *recent* rays
        ray_sets_by_marker_id: dict[int, list[MarkerRaySet]] = dict()
        for marker_key, marker_ray_set in self._marker_rayset_by_marker_key.items():
            if (now_timestamp - marker_ray_set.image_timestamp).total_seconds() > \
               POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS:
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
            ray_sets_by_marker_id[marker_id] = ray_set_list[0:MAXIMUM_RAY_COUNT_FOR_INTERSECTION]

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
                    maximum_distance=INTERSECTION_MAXIMUM_DISTANCE)
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
            if target_id == str(self._reference_target.marker_id):
                continue  # everything is expressed relative to the reference...
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
                # TODO: Re-enable when we are ready to do more advanced alpha-beta pose differentiation
                # position, orientation = self._estimate_target_pose_from_ray_set_and_single_marker_id(
                #     ray_set=ray_set,
                #     target_id=target_id,
                #     target=target)
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
                    termination_iteration_count=ITERATIVE_CLOSEST_POINT_TERMINATION_ITERATION_COUNT,
                    termination_delta_translation=ITERATIVE_CLOSEST_POINT_TERMINATION_TRANSLATION,
                    termination_delta_rotation_radians=ITERATIVE_CLOSEST_POINT_TERMINATION_ROTATION_RADIANS,
                    termination_mean_point_distance=ITERATIVE_CLOSEST_POINT_TERMINATION_MEAN_POINT_DISTANCE,
                    termination_rms_point_distance=ITERATIVE_CLOSEST_POINT_TERMINATION_RMS_POINT_DISTANCE)

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
                     self._target_depths_by_target_depth_key[target_depth_key]])) + POSE_SINGLE_CAMERA_DEPTH_CORRECTION
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
