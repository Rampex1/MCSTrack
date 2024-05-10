from src.board_builder.structures.pose_location import \
    PoseLocation

from src.common.structures import \
    Matrix4x4

from typing import Final

virtual_cube = {
    6: [4,4,0],
    1: [8,4,-4],
    2: [4,4,-8],
    3: [0,4,-4],
    4: [4,8,-4],
    5: [4,0,-4]
}



class Pose:
    def __init__(self, target_id, object_to_reference_matrix, solver_timestamp_utc_iso8601):
        self.target_id = target_id
        self.object_to_reference_matrix = object_to_reference_matrix
        self.solver_timestamp_utc_iso8601 = solver_timestamp_utc_iso8601

# List of transformation matrices, indexed by marker ID for clarity
import numpy as np

transformation_matrices = {
    
    1: np.array([[ 0.0,  0.0,  1.0,  4.0],
                 [ 0.0,  1.0,  0.0,  0.0],
                 [-1.0,  0.0,  0.0, -4.0],
                 [ 0.0,  0.0,  0.0,  1.0]]),

    2: np.array([[-1.0,  0.0,  0.0,  0.0],
                 [ 0.0,  1.0,  0.0,  0.0],
                 [ 0.0,  0.0, -1.0, -8.0],
                 [ 0.0,  0.0,  0.0,  1.0]]),

    3: np.array([[ 0.0,  0.0, -1.0, -4.0],
                 [ 0.0,  1.0,  0.0,  0.0],
                 [ 1.0,  0.0,  0.0, -4.0],
                 [ 0.0,  0.0,  0.0,  1.0]]),

    4: np.array([[ 1.0,  0.0,  0.0,  0.0],
                 [ 0.0,  0.0,  1.0,  4.0],
                 [ 0.0, -1.0,  0.0, -4.0],
                 [ 0.0,  0.0,  0.0,  1.0]]),

    5: np.array([[ 1.0,  0.0,  0.0,  0.0],
                 [ 0.0,  0.0, -1.0, -4.0],
                 [ 0.0,  1.0,  0.0, -4.0],
                 [ 0.0,  0.0,  0.0,  1.0]]),

    6: np.array([[ 1.0,  0.0,  0.0,  0.0],
                 [ 0.0,  1.0,  0.0,  0.0],
                 [ 0.0,  0.0,  1.0,  0.0],
                 [ 0.0,  0.0,  0.0,  1.0]]),
}


REFERENCE_MARKER_ID: Final[int] = 0
TARGET_MARKER_ID: Final[int] = 1
MARKER_SIZE_MM: Final[float] = 10.0

target_markers_list = []  # Keep track of what markers have appeared at least once

size_of_matrix = 0
relationship_matrix = [[None for _ in range(size_of_matrix)] for _ in range(size_of_matrix)]  # Matrix to record Transforms between markers  # Matrix to record Transforms between markers


def expand_matrix(matrix):
    size = len(matrix) + 1
    new_matrix = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size - 1):
        for j in range(size - 1):
            new_matrix[i][j] = matrix[i][j]
    return new_matrix

def calculate_relative_transform(T1, T2):
    T1_inv = np.linalg.inv(T1)

    # Compute the relative transformation matrix
    relative_T = np.dot(T1_inv, T2)

    return relative_T

def to_numpy_array(list):  # Assuming list is pose.object_to_reference_matrix, so 4x4
    lst = []
    for i in range(4):
        for j in range(4):
            lst.append(list[i,j])

    return np.array(lst).reshape(4,4)

def estimate_reference_to_invisible(T_AB, T_BC):
    T_AC = np.dot(T_AB, T_BC)
    return T_AC


### START ###
target_poses = [
    Pose(target_id=id, object_to_reference_matrix=Matrix4x4.from_numpy_array(matrix), solver_timestamp_utc_iso8601="now")
    for id, matrix in transformation_matrices.items()
]

# Output for verification
for pose in target_poses:
    target_markers_list.append(pose.target_id)
    relationship_matrix = expand_matrix(relationship_matrix)
    size_of_matrix += 1

### SOLVE POSE ###

for pose in target_poses:
    # R R R T
    # R R R T
    # R R R T
    # 0 0 0 1

    pose_matrix = to_numpy_array(pose.object_to_reference_matrix)

    for other_pose in target_poses:
        if other_pose != pose:

            other_pose_matrix = to_numpy_array(other_pose.object_to_reference_matrix)

            relative_position = calculate_relative_transform(pose_matrix, other_pose_matrix)

            print("RELATIVE", pose.target_id, other_pose.target_id)
            print(relative_position)

            matrix_entry = relationship_matrix[target_markers_list.index(pose.target_id)][target_markers_list.index(other_pose.target_id)]


            if not matrix_entry:  # Create a new object
                new_pose_location = PoseLocation()
                new_pose_location.add_matrix(relative_position)
                relationship_matrix[target_markers_list.index(pose.target_id)][
                    target_markers_list.index(other_pose.target_id)] = new_pose_location

            else:  # Add data
                relationship_matrix[target_markers_list.index(pose.target_id)][
                    target_markers_list.index(other_pose.target_id)].add_matrix(relative_position)

visible_markers = [1]

### ID IS NOT IN FRAME ###
for marker in target_markers_list:
    if marker not in visible_markers:
        marker_index = target_markers_list.index(marker)
        estimated_pose_location = PoseLocation()
        for j in range(len(target_markers_list) - 1):
            if relationship_matrix[j][marker_index] and target_markers_list[j] in visible_markers:
                T_AC = estimate_reference_to_invisible(transformation_matrices[target_markers_list[j]], relationship_matrix[j][marker_index].get_matrix())
                estimated_pose_location.add_matrix(T_AC)

        marker_position = estimated_pose_location.get_matrix()
        rounded_matrix = np.round(marker_position, decimals=3)

        print(f"The position of marker {marker} is {rounded_matrix}")

