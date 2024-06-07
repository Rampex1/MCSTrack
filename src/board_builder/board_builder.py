from src.common.structures import \
    IntrinsicParameters, \
    Matrix4x4, \
    Pose
from src.pose_solver.structures import \
    MarkerCorners, \
    TargetMarker, \
    MarkerRaySet, \
    PoseData, \
    Target, \
    PoseSolverParameters
from src.pose_solver.util import \
    transformation_image_to_opengl
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

class BoardBuilder2:
    """
    Class containing the actual "solver" logic, kept separate from the API.
    """

    _intrinsics_by_detector_label: dict[str, IntrinsicParameters]
    _targets: dict[uuid.UUID, Target]

    _marker_corners_since_update: list[MarkerCorners]

    _marker_rayset_by_marker_key: dict[MarkerKey, MarkerRaySet]

    _poses_by_target_id: dict[uuid.UUID, PoseData]
    _poses_by_detector_label: dict[str, Matrix4x4]

    def __init__(self):

        self._intrinsics_by_detector_label = dict()
        self._parameters = PoseSolverParameters()
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
            self._parameters.POSE_DETECTOR_DENOISE_LIMIT_AGE_SECONDS,
            self._parameters.POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS,
            self._parameters.POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS,
            self._parameters.POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS,
            self._parameters.POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS])

        self._image_point_sets_by_image_key = dict()

    def add_marker_corners(
        self,
        detected_corners: list[MarkerCorners]
    ) -> None:
        self._marker_corners_since_update += detected_corners

    def get_poses(
        self
    ) -> list[Pose]:
        detector_poses: list[Pose] = [
            Pose(
                target_id=detector_label,
                object_to_reference_matrix=pose,
                solver_timestamp_utc_iso8601=str(datetime.datetime.utcnow().isoformat()))
            for detector_label, pose in self._poses_by_detector_label.items()]
        return detector_poses

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

        self._image_point_sets_by_image_key = image_point_sets_by_image_key
        self._marker_corners_since_update.clear()

    def estimate_detector_pose_relative_to_reference(self):
        image_point_set_keys_with_reference_visible: list[ImagePointSetsKey] = list()
        for image_point_sets_key, image_point_sets in self._image_point_sets_by_image_key.items():
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
                [-half_width, half_width, 0.0],
                [half_width, half_width, 0.0],
                [half_width, -half_width, 0.0],
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

