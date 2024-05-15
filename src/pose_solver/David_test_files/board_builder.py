from cv2 import aruco
import numpy as np

from src.pose_solver.pose_solver import \
    ImagePointSetsKey, \
    MarkerKey, \
    CornerSetReference, \
    TargetDepthKey, \
    TargetDepth, \
    PoseExtrapolationQuality, \
    PoseSolver

from src.pose_solver.structures import \
    MarkerCorners, \
    TargetMarker, \
    Target

from src.common.structures import \
    IntrinsicParameters

from src.pose_solver.David_test_files.pose_location import \
    PoseLocation

import cv2
import cv2.aruco
import datetime
import numpy
from scipy.spatial.transform import Rotation
from typing import Callable, Final, Optional, TypeVar
import uuid

REFERENCE_MARKER_ID: Final[int] = 0
TARGET_MARKER_ID: Final[int] = 1
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

marker_corners_dict = {}  # Keep track of the location of the corners
target_markers_list = []  # Keep track of what markers have appeared at least once
target_markers_list_uuid = []

size_of_matrix = 0
relationship_matrix = np.zeros((size_of_matrix, size_of_matrix))  # Matrix to record Transforms between markers


def calculate_relative_position(rvec1, tvec1, rvec2, tvec2):
    # Convert rotation vectors to rotation matrices
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    # Create transformation matrices
    T1 = np.hstack((R1, tvec1))
    T2 = np.hstack((R2, tvec2))
    T1 = np.vstack((T1, [0, 0, 0, 1]))
    T2 = np.vstack((T2, [0, 0, 0, 1]))

    # Calculate the relative transformation matrix
    T1_inv = np.linalg.inv(T1)
    T_relative = np.dot(T1_inv, T2)

    # Extract relative rotation and translation
    relative_rotation_matrix = T_relative[0:3, 0:3]
    relative_translation_vector = T_relative[0:3, 3]
    relative_rotation_vector, _ = cv2.Rodrigues(relative_rotation_matrix)

    # Output the results
    return relative_rotation_vector, relative_translation_vector


def detect_aruco_from_camera(pose_solver):
    global size_of_matrix, relationship_matrix

    ### OPENCV SETUP ###
    cap = cv2.VideoCapture(1)  # Camera 1 for David's computer

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters_create()

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ### DETECTING CORNERS ###
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            ### ADD TARGET MARKER ###
            for marker_id in range(len(ids)):
                if ids[marker_id][0] not in target_markers_list and ids[marker_id] != REFERENCE_MARKER_ID:
                    target_markers_list.append(ids[marker_id][0])
                    target_marker_diameter = MARKER_SIZE_MM

                    ### EXPAND MATRIX ###
                    original_matrix = relationship_matrix
                    size_of_matrix += 1
                    expanded_matrix = np.zeros(
                        (size_of_matrix, size_of_matrix), dtype=original_matrix.dtype)
                    expanded_matrix[:size_of_matrix - 1, :size_of_matrix - 1] = original_matrix
                    relationship_matrix = expanded_matrix

                    marker_uuid = pose_solver.try_add_target_marker(ids[marker_id][0], target_marker_diameter)
                    target_markers_list_uuid.append(marker_uuid)


            ### ADD CORNERS ###
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label=DETECTOR_GREEN_NAME,
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),
                    timestamp=datetime.datetime.now()
                )
                pose_solver.add_marker_corners([marker_corners])

                marker_corners_dict[int(ids[i][0])] = corner[0].tolist()

            ### SOLVE POSE ###
            pose_solver.update()
            detector_poses, target_poses = pose_solver.get_poses()

            camera_matrix = np.array([
                [DETECTOR_GREEN_INTRINSICS.focal_length_x_px, 0, DETECTOR_GREEN_INTRINSICS.optical_center_x_px],
                [0, DETECTOR_GREEN_INTRINSICS.focal_length_y_px, DETECTOR_GREEN_INTRINSICS.optical_center_y_px],
                [0, 0, 1]], dtype=float)
            dist_coeffs = np.array(
                DETECTOR_GREEN_INTRINSICS.radial_distortion_coefficients + DETECTOR_GREEN_INTRINSICS.tangential_distortion_coefficients)

            for pose in target_poses:
                # R R R T
                # R R R T
                # R R R T
                # 0 0 0 1

                matrix_values = pose.object_to_reference_matrix.values
                matrix_4x4 = np.array(matrix_values).reshape(4, 4)
                rvec, _ = cv2.Rodrigues(matrix_4x4[:3, :3])
                tvec = matrix_4x4[:3, 3]
                np_array_tvec = []
                for i in tvec:
                    np_array_tvec.append([i])

                for other_pose in target_poses:
                    if other_pose != pose:
                        other_matrix_values = other_pose.object_to_reference_matrix.values
                        other_matrix_4x4 = np.array(other_matrix_values).reshape(4, 4)
                        other_rvec, _ = cv2.Rodrigues(other_matrix_4x4[:3, :3])
                        other_tvec = other_matrix_4x4[:3, 3]
                        other_np_array_tvec = []
                        for i in other_tvec:
                            other_np_array_tvec.append([i])

                        relative_position = calculate_relative_position(rvec, other_rvec, np.array(np_array_tvec),
                                                                        np.array(other_np_array_tvec))

                        #print("POSE1", pose)
                        #print("Pose2", other_pose)
                        #print("RELATIVE", relative_position)

                        matrix_entry = relationship_matrix[target_markers_list_uuid.index(pose.target_id)][target_markers_list_uuid.index(other_pose.target_id)]
                        print(matrix_entry)

                        if matrix_entry == 0:  # Create a new object

                            new_pose_location = PoseLocation()
                            new_pose_location.add_RVEC(relative_position[0])
                            new_pose_location.add_TVEC(relative_position[1])
                            relationship_matrix[target_markers_list_uuid.index(pose.target_id)][
                                target_markers_list_uuid.index(other_pose.target_id)] = new_pose_location

                        """

                        else: # Add data
                            relationship_matrix[target_markers_list_uuid.index(pose.target_id)][
                                target_markers_list_uuid.index(other_pose.target_id)].add_RVEC(relative_position[0])
                            relationship_matrix[target_markers_list_uuid.index(pose.target_id)][
                                target_markers_list_uuid.index(other_pose.target_id)].add_TVEC(relative_position[1])

                        print("RESULT", relationship_matrix)
                        """

                        # If element at pose, other_pose = 0, create new pose_location
                        # add RVEC, TVEC

                # NOTE: tvec is rescaled so it can be shown on screen
                tvec[0] /= 500  # 1100
                tvec[1] /= 900
                tvec[2] = 2

                frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        ### DISPLAY FRAME ###
        cv2.imshow('Frame with ArUco markers', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


pose_solver = PoseSolver()

### SET INTRINSIC PARAMETERS ###
pose_solver.set_intrinsic_parameters(DETECTOR_GREEN_NAME, DETECTOR_GREEN_INTRINSICS)

### SET REFERENCE TARGET ###
reference_target: Target = TargetMarker(
    marker_id=REFERENCE_MARKER_ID,
    marker_size=MARKER_SIZE_MM)
pose_solver.set_reference_target(reference_target)

detect_aruco_from_camera(pose_solver)

print("Target Marker List:", target_markers_list)
print("Target Marker List UUID:", target_markers_list_uuid)

### PSEUDOCODE FOR POSE TRANSPOSE MATRIX ###

"""
Assumptions for now: 2 cameras, reference board that DOESN'T move (we use reference as origin), target markers that can move

Main Idea: Create a matrix where the index i,j contains information about the relationship between markers i and j
[[all RVEC], RVEC_mean = 0, [all TVEC], TVEC_mean = 0, frame appearance % (optional)] # Could construct an object for this


X   M1  M2  M3  M4
M1  X   
M2      X
M3          X
M4              X

Start with 0x0 matrix
markers_that_have_appeared = []

for each frame:
    for each detected marker "i":
        # 1. Update the matrix dimensions if necessary
        if marker not in markers_that_have_appeared:
            matrix_size += 1
            markers_that_have_appeared.append marker

        # 2. Update the matrix inputs
        for each other marker "j":
            new_relative_pose = compute_relative_pose(RVEC1, RVEC2, TVEC1, TVEC2) # Output of the form [RVEC, TVEC]

            # Fill in matrix for RVEC
            if matrix[i][j][0].length == 0:  # No previous input for RVEC
                matrix[i][j][0] = new_relative_pose[0]
                matrix[i][j][1] = new_relative_pose
            else:  # Previous data exists, we take the mean
                matrix[i][j][0].append(new_relative_pose)
                matrix[i][j][1] = mean(matrix[i][j][0])

            # Fill in matrix for TVEC
            if matrix[i][j][2].length == 0:  # No previous input for RVEC
                matrix[i][j][2] = new_relative_pose[1]
                matrix[i][j][3] = new_relative_pose
            else:  # Previous data exists, we take the mean
                matrix[i][j][2].append(new_relative_pose)

        # By this point, the algorithm should be able to start constructing a table of pose relations between markers i,j
        # If two markers i,j have matrix[i][j][1] = 0 or matrix[i][j][3] = 0, it means that the two markers have yet to appear in the same frame together

        # 3. Find the coordinate of the CENTER of all markers
        for each markers_that_have_appeared k:
            if marker_on_screen:
                use_pose_solver
            if marker_not_on_screen:
                use_matrix

        use_matrix:
            k_position = []
            for matrix[i][k]: # Each input of row k
                if mean != 0 and marker i is visible:
                    k_position.append(Transform of i from camera + transform from matrix[i][k]
            estimated_k_position = mean(k_position)

        # By this point, we have a transform relative to the reference for each marker that has appeared at least once

        # 4. Find the corners
        # If we know the size of a marker, we can find the relation size of marker / RVEC in x, and use these proportions
        # to find the position of its corners
"""
