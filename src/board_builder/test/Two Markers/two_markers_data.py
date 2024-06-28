import numpy as np
import two_markers_corners
from src.board_builder.test import setup_data

class TwoMarkersAccuracy:
    def __init__(self, TA, TB):
        # Define the transformation matrix T for Detector A
        self.TA = TA
        self.TB = TB

    def transform_to_detector_A(self, corners_world):
        # Add the homogeneous coordinate to the corner data
        corners_world_homogeneous = np.hstack([corners_world, np.ones((corners_world.shape[0], 1))])

        # Apply the transformation matrix
        corners_detector_A_homogeneous = self.TA @ corners_world_homogeneous.T

        # Convert back from homogeneous coordinates to 3D coordinates
        corners_detector_A = corners_detector_A_homogeneous[:3].T

        return corners_detector_A



two_markers_accuracy = TwoMarkersAccuracy(setup_data.TA, setup_data.TA)

corners_marker_0_detector_A = two_markers_accuracy.transform_to_detector_A(two_markers_corners.corners_marker_0)
corners_marker_1_detector_A = two_markers_accuracy.transform_to_detector_A(two_markers_corners .corners_marker_1)

print("Marker 0 Corners in Detector A's Coordinate System:")
print(corners_marker_0_detector_A)

print("\nMarker 1 Corners in Detector A's Coordinate System:")
print(corners_marker_1_detector_A)