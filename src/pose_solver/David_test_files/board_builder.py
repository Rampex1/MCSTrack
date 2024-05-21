import cv2
from cv2 import aruco
import numpy as np
import datetime

from typing import Final
from src.common.structures import \
    IntrinsicParameters
from src.pose_solver.pose_solver import \
    PoseSolver
from src.pose_solver.structures import \
    MarkerCorners, \
    TargetMarker, \
    Target
from src.pose_solver.David_test_files.pose_location import \
    PoseLocation


REFERENCE_MARKER_ID: Final[int] = 0
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
camera_matrix = np.array([
                [DETECTOR_GREEN_INTRINSICS.focal_length_x_px, 0, DETECTOR_GREEN_INTRINSICS.optical_center_x_px],
                [0, DETECTOR_GREEN_INTRINSICS.focal_length_y_px, DETECTOR_GREEN_INTRINSICS.optical_center_y_px],
                [0, 0, 1]], dtype=float)
dist_coeffs = np.array(
    DETECTOR_GREEN_INTRINSICS.radial_distortion_coefficients + DETECTOR_GREEN_INTRINSICS.tangential_distortion_coefficients)




matrix_size = 0
relative_pose_matrix = [[None for _ in range(matrix_size)] for _ in range(matrix_size)]  # Tracks relative pose between markers
target_marker_to_uuid = {}  # Maps target markers and their uuid
index_to_target_marker = {}  # Maps index [0, N] to the target_marker

def expand_matrix(matrix):
    """ Adds one row and one column to the matrix and initializes them to None """

    size = len(matrix) + 1
    new_matrix = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size - 1):
        for j in range(size - 1):
            new_matrix[i][j] = matrix[i][j]
    return new_matrix

def calculate_relative_transform(T1, T2):
    """ Given transform T1 from reference to marker 1, and transfrom T2 from reference to marker 2, calculate the
    transform from T1 to T2"""

    T1_inv = np.linalg.inv(T1)

    # Compute the relative transformation matrix
    relative_T = np.dot(T1_inv, T2)

    return relative_T

def find_matrix_input_index(pose_uuid, other_pose_uuid):
    """ Given two pose uuids, return their index location in the relative pose matrix """
    pose_id = -1
    other_pose_id = -1
    pose_index = -1
    other_pose_index = -1

    for id in target_marker_to_uuid:
        if target_marker_to_uuid[id] == pose_uuid:
            pose_id = id
        if target_marker_to_uuid[id] == other_pose_uuid:
            other_pose_id = id

    if pose_id != -1 and other_pose_id != -1:
        for index in index_to_target_marker:
            if index_to_target_marker[index] == pose_id:
                pose_index = index
            if index_to_target_marker[index] == other_pose_id:
                other_pose_index = index
        return pose_index, other_pose_index

    return None

def estimate_reference_to_not_visible(T_AB, T_BC):
    T_AC = np.dot(T_AB, T_BC)
    return T_AC



def board_builder(pose_solver):
    global matrix_size, relative_pose_matrix

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
            visible_markers = []  # List of markers that are visible in a specfic frame

            ### ADD TARGET MARKER ###
            for marker_id in range(len(ids)):
                visible_markers.append(ids[marker_id][0])
                if ids[marker_id][0] not in target_marker_to_uuid and ids[marker_id][0] != REFERENCE_MARKER_ID:
                    target_marker_diameter = MARKER_SIZE_MM
                    marker_uuid = pose_solver.try_add_target_marker(ids[marker_id][0], target_marker_diameter)
                    target_marker_to_uuid[ids[marker_id][0]] = marker_uuid
                    index_to_target_marker[matrix_size] = ids[marker_id][0]

                    ### EXPAND MATRIX ###
                    relative_pose_matrix = expand_matrix(relative_pose_matrix)
                    matrix_size += 1

            ### ADD CORNERS ###
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label=DETECTOR_GREEN_NAME,
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),
                    timestamp=datetime.datetime.now()
                )
                pose_solver.add_marker_corners([marker_corners])


            ### SOLVE POSE ###
            pose_solver.update()
            detector_poses, target_poses = pose_solver.get_poses()


            for pose in target_poses:
                # R R R T
                # R R R T
                # R R R T
                # 0 0 0 1

                print("POSE", pose.target_id, pose.object_to_reference_matrix.values)
                pose_values = pose.object_to_reference_matrix.values
                pose_matrix = np.array(pose_values).reshape(4, 4)

                for other_pose in target_poses:
                    if other_pose != pose:
                        other_matrix_values = other_pose.object_to_reference_matrix.values
                        other_pose_matrix = np.array(other_matrix_values).reshape(4, 4)

                        relative_transform = calculate_relative_transform(pose_matrix, other_pose_matrix)

                        matrix_index = find_matrix_input_index(pose.target_id, other_pose.target_id)


                        if not relative_pose_matrix[matrix_index[0]][matrix_index[1]]:  # Create a new object
                            new_pose_location = PoseLocation()
                            new_pose_location.add_matrix(relative_transform)
                            relative_pose_matrix[matrix_index[0]][matrix_index[1]] = new_pose_location

                        else:
                            relative_pose_matrix[matrix_index[0]][matrix_index[1]].add_matrix(relative_transform)


            ### ID IS NOT IN FRAME ###
            for marker in list(target_marker_to_uuid.keys()):  # Marker is an id
                if marker not in visible_markers:
                    estimated_pose_location = PoseLocation()
                    for other_marker in target_poses:  # Other marker is a pose

                        matrix_index = find_matrix_input_index(other_marker.target_id, target_marker_to_uuid[marker])

                        # Find the associated id
                        other_marker_id = -1
                        for id in target_marker_to_uuid:
                            if target_marker_to_uuid[id] == other_marker.target_id:
                                other_marker_id = id


                        if relative_pose_matrix[matrix_index[0]][matrix_index[1]] and other_marker_id in visible_markers:
                            # Given ref to visible (T_AB) and visible to hidden (T_BC) find ref to hidden (T_AC)
                            T_AB = other_marker.object_to_reference_matrix.values
                            T_AB = np.reshape(T_AB, (4, 4))
                            T_BC = relative_pose_matrix[matrix_index[0]][matrix_index[1]].get_TMatrix()
                            T_AC = estimate_reference_to_not_visible(T_AB, T_BC)
                            estimated_pose_location.add_matrix(T_AC)

                    marker_position = estimated_pose_location.get_TMatrix()
                    rounded_matrix = np.round(marker_position, decimals=3)

                    print(f"The position of marker {target_marker_to_uuid[marker]} is {rounded_matrix}")




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

board_builder(pose_solver)

print(target_marker_to_uuid)
print(index_to_target_marker)
print(relative_pose_matrix)
