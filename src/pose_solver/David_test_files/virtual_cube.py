from cv2 import aruco
import numpy as np

from src.pose_solver.pose_solver import \
    PoseSolver

from src.pose_solver.structures import \
    MarkerCorners, \
    TargetMarker, \
    Target

from src.common.structures import \
    IntrinsicParameters

from src.pose_solver.David_test_files.pose_location import \
    PoseLocation

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


virtual_cube = {
    6: [[1,1,0], [1,7,0], [7,7,0], [7,1,0]],
    1: [[1,8,-1], [1,8,-7], [7,8,-7], [7,8,-1]],
    2: [[1,7,-8], [1,1,-8], [7,1,-8], [7,7,-8]],
    3: [[1,0,-7], [1,0,-1], [7,0,-1], [7,0,-7]],
    4: [[0,1,-7], [0,7,-7], [0,7,-1], [0,1,-1]],
    5: [[8,1,-1], [8,7,-1], [8,7,-7], [8,1,-7]]
}

class Pose:
    def __init__(self, target_id, object_to_reference_matrix, solver_timestamp_utc_iso8601):
        self.target_id = target_id
        self.object_to_reference_matrix = object_to_reference_matrix
        self.solver_timestamp_utc_iso8601 = solver_timestamp_utc_iso8601

# List of transformation matrices, indexed by marker ID for clarity
transformation_matrices = {
    1: np.array([[ 6.123234e-17,  1.000000e+00,  0.000000e+00,  0.000000e+00],
                 [-1.000000e+00,  6.123234e-17,  0.000000e+00,  4.000000e+00],
                 [ 0.000000e+00,  0.000000e+00,  1.000000e+00, -4.000000e+00],
                 [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]),

    2: np.array([[-1.0000000e+00,  1.2246468e-16,  0.0000000e+00,  0.0000000e+00],
                 [-1.2246468e-16, -1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                 [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00, -8.0000000e+00],
                 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]),

    3: np.array([[ 6.123234e-17,  1.000000e+00,  0.000000e+00,  0.000000e+00],
                 [-1.000000e+00,  6.123234e-17,  0.000000e+00, -4.000000e+00],
                 [ 0.000000e+00,  0.000000e+00,  1.000000e+00, -4.000000e+00],
                 [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]),

    4: np.array([[ 1., -0.,  0., -4.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0.,  1., -4.],
                 [ 0.,  0.,  0.,  1.]]),

    5: np.array([[ 1., -0.,  0.,  4.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0.,  1., -4.],
                 [ 0.,  0.,  0.,  1.]]),

    6: np.array([[ 1., -0.,  0.,  0.],
                 [ 0.,  1.,  0.,  0.],
                 [ 0.,  0.,  1.,  0.],
                 [ 0.,  0.,  0.,  1.]])
}

# Create target_poses list with new Pose objects
target_poses = [
    Pose(target_id=id, object_to_reference_matrix=Matrix4x4.from_numpy_array(matrix), solver_timestamp_utc_iso8601="now")
    for id, matrix in transformation_matrices.items()
]

# Output for verification
for pose in target_poses:
    print(f"Target ID: {pose.target_id}, Matrix: {pose.object_to_reference_matrix}, Timestamp: {pose.solver_timestamp_utc_iso8601}\n")

REFERENCE_MARKER_ID: Final[int] = 0
TARGET_MARKER_ID: Final[int] = 1
MARKER_SIZE_MM: Final[float] = 10.0

target_markers_list = []  # Keep track of what markers have appeared at least once
target_markers_list_uuid = []

size_of_matrix = 0
relationship_matrix = [[None for _ in range(size_of_matrix)] for _ in range(size_of_matrix)]  # Matrix to record Transforms between markers  # Matrix to record Transforms between markers


def expand_matrix(matrix):
    size = len(matrix) + 1
    new_matrix = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size - 1):
        for j in range(size - 1):
            new_matrix[i][j] = matrix[i][j]
    return new_matrix

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



pose_solver = PoseSolver()

### SET REFERENCE TARGET ###
reference_target: Target = TargetMarker(
    marker_id=REFERENCE_MARKER_ID,
    marker_size=MARKER_SIZE_MM)
pose_solver.set_reference_target(reference_target)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
parameters = aruco.DetectorParameters_create()
visible_markers = [6,1,2,3,4,5]


### ADD TARGET MARKER ###
for marker_id in range(len(visible_markers)):
    relationship_matrix = expand_matrix(relationship_matrix)
    size_of_matrix += 1
    marker_uuid = pose_solver.try_add_target_marker(visible_markers[marker_id], MARKER_SIZE_MM)
    target_markers_list_uuid.append(marker_uuid)

### ADD CORNERS ###
for i in range(len(visible_markers)):
    marker_corners = MarkerCorners(
        detector_label= "default_camera",
        marker_id=visible_markers[i],
        points=virtual_cube[visible_markers[i]],
        timestamp=datetime.datetime.now()
    )
    pose_solver.add_marker_corners([marker_corners])



### SOLVE POSE ###

print(target_poses)

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

    #print("POSE", pose.target_id)
    #print(f"Marker {pose.target_id}: TVEC = {tvec}, RVEC = {rvec}")

    """
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

            matrix_entry = relationship_matrix[target_markers_list_uuid.index(pose.target_id)][target_markers_list_uuid.index(other_pose.target_id)]

            if not matrix_entry:  # Create a new object
                new_pose_location = PoseLocation()
                new_pose_location.add_RVEC(relative_position[0])
                new_pose_location.add_TVEC(relative_position[1])
                relationship_matrix[target_markers_list_uuid.index(pose.target_id)][
                    target_markers_list_uuid.index(other_pose.target_id)] = new_pose_location

            else: # Add data
                relationship_matrix[target_markers_list_uuid.index(pose.target_id)][
                    target_markers_list_uuid.index(other_pose.target_id)].add_RVEC(relative_position[0])
                relationship_matrix[target_markers_list_uuid.index(pose.target_id)][
                    target_markers_list_uuid.index(other_pose.target_id)].add_TVEC(relative_position[1])

"""

"""
            ### ID IS NOT IN FRAME ###
            for marker in target_markers_list:
                if marker not in visible_markers:
                    marker_index = target_markers_list.index(marker)
                    estimated_pose_location = PoseLocation()
                    for j in range(len(target_markers_list) - 1):
                        if relationship_matrix[j][marker_index]:
                            # We know reference to visible marker and visible marker to hidden marker
                            # Find reference to hidden marker
                            Ra_matrix, _ = cv2.Rodrigues(rvec)
                            Rb_matrix, _ = cv2.Rodrigues(relationship_matrix[j][marker_index].get_RVEC())

                            R_origin_to_b = Ra_matrix @ Rb_matrix
                            T_origin_to_b = Ra_matrix @ tvec + relationship_matrix[j][marker_index].get_TVEC()

                            R_origin_to_b, _ = cv2.Rodrigues(R_origin_to_b)

                            estimated_pose_location.add_RVEC(R_origin_to_b)
                            estimated_pose_location.add_TVEC(T_origin_to_b)

                    print(f"Estimated pose location for marker {marker} is TVEC = {estimated_pose_location.get_TVEC()} "
                          f"and RVEC = {estimated_pose_location.get_RVEC()}")




print("------------------------------------------------------------------------------------------------------------------")
print("Target Marker List:", target_markers_list)
print("Target Marker List UUID:", target_markers_list_uuid)
print("Relationship Matrix:", relationship_matrix)
"""
