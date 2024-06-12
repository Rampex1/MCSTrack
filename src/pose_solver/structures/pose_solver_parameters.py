from pydantic import BaseModel, Field
import cv2
import cv2.aruco

class PoseSolverParameters(BaseModel):
    MAXIMUM_RAY_COUNT_FOR_INTERSECTION: int = Field(2)
    POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS: float = Field(0.1)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_MINIMUM_SURFACE_NORMAL_ANGLE_DEGREES: float = Field(15.0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS: float = Field(1.0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_MAXIMUM_ORDER: int = Field(0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_ANGLE_DEGREES: float = Field(15.0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_DISTANCE: float = Field(15.0, description="millimeters")
    POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS: float = Field(0.8)
    POSE_SINGLE_CAMERA_NEAREST_LIMIT_ANGLE_DEGREES: float = Field(15.0)
    POSE_SINGLE_CAMERA_NEAREST_LIMIT_DISTANCE: float = Field(15.0)
    POSE_SINGLE_CAMERA_REPROJECTION_ERROR_FACTOR_BETA_OVER_ALPHA: float = Field(1.0)
    POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS: float = Field(0.4)
    # TODO: Is this next one detector-specific?
    POSE_SINGLE_CAMERA_DEPTH_CORRECTION: float = Field(-7.5, description="millimeters, observed tendency to overestimate depth.")
    POSE_DETECTOR_DENOISE_LIMIT_AGE_SECONDS: float = Field(1.0)
    INTERSECTION_MAXIMUM_DISTANCE: float = Field(10.0, description="millimeters")
    ITERATIVE_CLOSEST_POINT_TERMINATION_ITERATION_COUNT: int = Field(50)
    ITERATIVE_CLOSEST_POINT_TERMINATION_TRANSLATION: float = Field(0.005, description="millimeters")
    ITERATIVE_CLOSEST_POINT_TERMINATION_ROTATION_RADIANS: float = Field(0.0005)
    ITERATIVE_CLOSEST_POINT_TERMINATION_MEAN_POINT_DISTANCE: float = Field(0.1, description="millimeters")
    ITERATIVE_CLOSEST_POINT_TERMINATION_RMS_POINT_DISTANCE: float = Field(0.1, description="millimeters")
    DENOISE_OUTLIER_DISTANCE_MILLIMETERS: float = Field(10.0)
    DENOISE_OUTLIER_ANGLE_DEGREES: float = Field(5.0)
    DENOISE_STORAGE_SIZE: int = Field(10)
    DENOISE_FILTER_SIZE: int = Field(7)
    DENOISE_REQUIRED_STARTING_STREAK: int = Field(3)
    ARUCO_MARKER_DICTIONARY_ENUM: int = Field(cv2.aruco.DICT_4X4_100)
    ARUCO_POSE_ESTIMATOR_METHOD: int = Field(cv2.SOLVEPNP_ITERATIVE)
    # SOLVEPNP_ITERATIVE works okay but is susceptible to optical illusions (flipping)
    # SOLVEPNP_P3P appears to return nan's on rare occasion
    # SOLVEPNP_SQPNP appears to return nan's on rare occasion
    # SOLVEPNP_IPPE_SQUARE does not seem to work very well at all, translation is much smaller than expected
