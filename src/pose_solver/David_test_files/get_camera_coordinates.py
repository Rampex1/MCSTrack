from cv2 import aruco

from src.pose_solver.pose_solver import \
    ImagePointSetsKey, \
    MarkerKey, \
    CornerSetReference, \
    TargetDepthKey, \
    TargetDepth, \
    PoseExtrapolationQuality, \
    PoseSolver

from src.pose_solver.structures import \
    MarkerCorners

import cv2
import cv2.aruco
import datetime
import numpy
from scipy.spatial.transform import Rotation
from typing import Callable, Final, Optional, TypeVar
import uuid


pose_solver = PoseSolver()

def detect_aruco_from_camera(pose_solver):
    # Open camera 1
    cap = cv2.VideoCapture(1)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Load the predefined dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

    # Initialize the detector parameters using default values
    parameters = aruco.DetectorParameters_create()

    # Dictionary to hold marker IDs and their corners
    marker_corners_dict = {}

    while True:

        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




        # Detect the markers in the frame
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Check if markers are found and draw them
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Find the marker corners each frame
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label='default_camera',  # Use a unique identifier for your camera
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),  # Convert each corner point to a list
                    timestamp=datetime.datetime.now()
                )
                pose_solver.add_marker_corners([marker_corners])  # Assuming a method to add corners exists
                marker_corners_dict[int(ids[i][0])] = corner[0].tolist()
                #print('default_camera', int(ids[i][0]), corner[0].tolist(), datetime.datetime.now())
                #pose_solver.update()
                print(marker_corners_dict)

                if len(marker_corners_dict) > 1:
                    first_marker_id = list(marker_corners_dict.keys())[0]
                    first_marker_topleft_corner = marker_corners_dict[first_marker_id][0]

                    # Draw vectors from the top-left corner of the first marker to every other marker's top-left corner
                    for key, value in marker_corners_dict.items():
                        if key != first_marker_id:
                            topleft_corner = value[0]
                            cv2.line(frame, tuple(map(int, first_marker_topleft_corner)),
                                     tuple(map(int, topleft_corner)), (0, 255, 0), 2)

        else:
            print("No markers detected.")




        # Display the resulting frame
        cv2.imshow('Frame with ArUco markers', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



detect_aruco_from_camera(pose_solver)