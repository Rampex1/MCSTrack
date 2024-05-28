from src.common import \
    client_identifier_from_connection, \
    EmptyResponse, \
    ErrorResponse
from src.detector import \
    Detector, \
    DetectorConfiguration
from src.detector.api import \
    GetCaptureDeviceResponse, \
    GetCaptureImageRequest, \
    GetCaptureImageResponse, \
    GetCapturePropertiesResponse, \
    GetDetectionParametersResponse, \
    GetMarkerSnapshotsRequest, \
    GetMarkerSnapshotsResponse, \
    SetCaptureDeviceRequest, \
    SetDetectionParametersRequest
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from fastapi_utils.tasks import repeat_every
import hjson
import logging
import os

from src.detector.implementations import AbstractMarkerInterface, ArucoMarker


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    detector_configuration_filepath: str = os.path.join(os.path.dirname(__file__), "..", "..", "data", "config.json")
    detector_configuration: DetectorConfiguration
    #camera_interface: AbstractCameraInterface
    marker_interface: AbstractMarkerInterface

    with open(detector_configuration_filepath, 'r') as infile:
        detector_configuration_file_contents: str = infile.read()
        detector_configuration_dict = hjson.loads(detector_configuration_file_contents)
        detector_configuration = DetectorConfiguration(**detector_configuration_dict)

    #camera_interface = Detector.USBWebcamWithOpenCV()
    marker_interface = ArucoMarker()
    
    detector = Detector(detector_configuration=detector_configuration,marker_interface=marker_interface)
    detector_app = FastAPI()

    # CORS Middleware
    origins = ["http://localhost"]
    detector_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @detector_app.get("/get_capture_device")
    async def get_capture_device() -> GetCaptureDeviceResponse:
        result: GetCaptureDeviceResponse = detector.get_capture_device()
        return result

    @detector_app.get("/get_capture_image")
    async def get_capture_image() -> GetCaptureImageResponse:
        result: GetCaptureImageResponse = detector.get_capture_image(
            request=GetCaptureImageRequest(format=".png"))
        image_bytes = base64.b64decode(result.image_base64)
        with open("test.png", "wb") as image_file:
            image_file.write(image_bytes)
        return result

    @detector_app.get("/get_capture_properties")
    async def get_capture_properties() -> GetCapturePropertiesResponse:
        result: GetCapturePropertiesResponse = detector.get_capture_properties()
        return result

    @detector_app.get("/get_detection_parameters")
    async def get_detection_parameters() -> GetDetectionParametersResponse | ErrorResponse:
        return detector.get_detection_parameters()

    @detector_app.post("/get_marker_snapshots")
    async def get_marker_snapshots(
        request: GetMarkerSnapshotsRequest
    ) -> GetMarkerSnapshotsResponse:
        return detector.get_marker_snapshots(
            request=request)

    @detector_app.post("/set_capture_device")
    async def set_capture_device(
        request: SetCaptureDeviceRequest
    ) -> EmptyResponse | ErrorResponse:
        return detector.set_capture_device(
            request=request)

    @detector_app.post("/set_detection_parameters")
    async def set_detection_parameters(
        request: SetDetectionParametersRequest
    ) -> EmptyResponse | ErrorResponse:
        return detector.set_detection_parameters(
            request=request)

    @detector_app.head("/start_capture")
    async def start_capture(
        http_request: Request
    ) -> None:
        client_identifier: str = client_identifier_from_connection(connection=http_request)
        detector.start_capture(client_identifier=client_identifier)

    @detector_app.head("/stop_capture")
    async def stop_capture(
        http_request: Request
    ) -> None:
        client_identifier: str = client_identifier_from_connection(connection=http_request)
        detector.stop_capture(client_identifier=client_identifier)

    @detector_app.websocket("/websocket")
    async def websocket_handler(websocket: WebSocket) -> None:
        await detector.websocket_handler(websocket=websocket)

    @detector_app.on_event("startup")
    @repeat_every(seconds=0.001)
    async def internal_update() -> None:
        await detector.internal_update()

    return detector_app


app = create_app()
