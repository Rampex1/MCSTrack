from src.common import \
    EmptyResponse, \
    ErrorResponse
from src.board_builder.board_builder import \
    BoardBuilder
from src.board_builder.board_builder_api import \
    BoardBuilderAPI, \
    BoardBuilderConfiguration
from src.board_builder.api import \
    SetIntrinsicParametersRequest, \
    LocateReferenceMarkersRequest, \
    CollectDataRequest, \
    BuildBoardRequest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
from fastapi_utils.tasks import repeat_every
import hjson
import logging
import os


logger = logging.getLogger(__name__)
board_builder = BoardBuilder()


def create_app() -> FastAPI:
    configuration_filepath: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "board_builder_config.json")
    configuration: BoardBuilderConfiguration
    with open(configuration_filepath, 'r') as infile:
        file_contents: str = infile.read()
        configuration_dict = hjson.loads(file_contents)
        configuration = BoardBuilderConfiguration(**configuration_dict)
    board_builder_api = BoardBuilderAPI(
        configuration=configuration,
        board_builder=board_builder)
    board_builder_app = FastAPI()

    # CORS Middleware
    origins = ["http://localhost"]
    board_builder_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @board_builder_app.post("/set_intrinsic_parameters")
    async def set_intrinsic_parameters(
        request: SetIntrinsicParametersRequest
    ) -> EmptyResponse | ErrorResponse:
        return board_builder_api.set_intrinsic_parameters(
            request=request)

    @board_builder_app.head("/start_capture")
    async def start_capture() -> None:
        board_builder_api.start_board_builder()

    @board_builder_app.head("/stop_capture")
    async def stop_capture() -> None:
        board_builder_api.stop_board_builder()

    @board_builder_app.post("/locate_reference_markers")
    async def locate_reference_markers(
        request: LocateReferenceMarkersRequest
    ) -> EmptyResponse | ErrorResponse:
        return board_builder_api.locate_reference_markers(
            request=request)

    @board_builder_app.post("/collect_data")
    async def collect_data(
        request: CollectDataRequest
    ) -> dict | ErrorResponse:
        return board_builder_api.collect_data(
            request=request)

    @board_builder_app.post("/build_board")
    async def build_board(
        request: BuildBoardRequest
    ) -> dict | ErrorResponse:
        return board_builder_api.build_board(
            request=request)

    @board_builder_app.websocket("/websocket")
    async def websocket_handler(websocket: WebSocket) -> None:
        await board_builder_api.websocket_handler(websocket=websocket)

    @board_builder_app.on_event("startup")
    @repeat_every(seconds=0.001)
    async def internal_update() -> None:
        await board_builder_api.internal_update()

    return board_builder_app


app = create_app()
