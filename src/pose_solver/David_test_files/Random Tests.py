import numpy as np


def compute_transformation(reference_points, target_points):
    # Calculate the translation needed to align the centroids
    ref_centroid = np.mean(reference_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    translation = target_centroid - ref_centroid

    # Assume simple rotation around Z-axis because all transformations are in-plane
    ref_vector = reference_points[1] - reference_points[0]  # from top-left to top-right
    target_vector = target_points[1] - target_points[0]
    ref_angle = np.arctan2(ref_vector[1], ref_vector[0])
    target_angle = np.arctan2(target_vector[1], target_vector[0])
    rotation_angle = target_angle - ref_angle

    # Create rotation matrix around Z-axis
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    # Create the full 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation

    return transform_matrix


# Reference coordinates (same as marker ID 6)
ref_coords = np.array([[1, 1, 0], [1, 7, 0], [7, 7, 0], [7, 1, 0]])

# Virtual cube definition
virtual_cube = {
    1: [[1, 8, -1], [1, 8, -7], [7, 8, -7], [7, 8, -1]],
    2: [[1, 7, -8], [1, 1, -8], [7, 1, -8], [7, 7, -8]],
    3: [[1, 0, -7], [1, 0, -1], [7, 0, -1], [7, 0, -7]],
    4: [[0, 1, -7], [0, 7, -7], [0, 7, -1], [0, 1, -1]],
    5: [[8, 1, -1], [8, 7, -1], [8, 7, -7], [8, 1, -7]],
    6: [[1, 1, 0], [1, 7, 0], [7, 7, 0], [7, 1, 0]]
}

transformations = {}
for id, coords in virtual_cube.items():
    if id != 0:  # Reference ID is set as 0, but using ID 6 for calculations
        transformations[id] = compute_transformation(ref_coords, np.array(coords))

# Output the transformation matrices
for id, matrix in transformations.items():
    print(f"Transformation Matrix for Marker {id}:\n{matrix}\n")
