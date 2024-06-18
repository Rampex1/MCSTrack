from typing import Final
from .camera_connection import CameraConnection
from src.common.structures import IntrinsicParameters
from pydantic import BaseModel, Field

OPENCV: Final[str] = "opencv"
PICAMERA: Final[str] = "picamera"

class DetectorConfiguration(BaseModel):
    """
    Top-level schema for Detector initialization data
    """
    serial_identifier: str = Field()
    camera_implementation: str = Field(default=OPENCV)
    camera_connection: CameraConnection = Field()
    camera_intrinsic_parameters: dict[str, IntrinsicParameters] | None = Field(default_factory=dict)
