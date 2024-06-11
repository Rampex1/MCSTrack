import cv2
import uuid
import logging
from typing import Final, Optional
import wx
import wx.grid
from cv2 import aruco

from .base_panel import BasePanel
from .feedback import ImagePanel
from .parameters import ParameterSelector, ParameterSpinboxFloat, ParameterSpinboxInteger

from src.calibrator.api import ListCalibrationDetectorResolutionsRequest
from src.detector.api import GetCapturePropertiesRequest
from src.board_builder.api import SetReferenceMarkerRequest
from src.board_builder import BoardBuilder

from src.common.structures import IntrinsicParameters
from src.connector import Connector
from src.common import (
    ErrorResponse,
    EmptyResponse,
    MCastRequestSeries,
    MCastResponse,
    MCastResponseSeries,
    StatusMessageSource
)

logger = logging.getLogger(__name__)

# TODO: ????
ACTIVE_PHASE_IDLE: Final[int] = 0
ACTIVE_PHASE_STARTING_CAPTURE: Final[int] = 1
ACTIVE_PHASE_STARTING_GET_RESOLUTIONS: Final[int] = 2
ACTIVE_PHASE_STARTING_LIST_INTRINSICS: Final[int] = 3  # This and next phase to be combined with modified API
ACTIVE_PHASE_STARTING_GET_INTRINSICS: Final[int] = 4
ACTIVE_PHASE_STARTING_FINAL: Final[int] = 5
ACTIVE_PHASE_STOPPING: Final[int] = 6


class BoardBuilderPanel(BasePanel):
    _connector: Connector

    _board_builder_selector: ParameterSelector
    _reference_marker_id_spinbox: ParameterSpinboxInteger
    _reference_marker_diameter_spinbox: ParameterSpinboxFloat
    _reference_target_submit_button: wx.Button

    _open_camera_button: wx.Button
    _locate_reference_button: wx.Button
    _collect_data_button: wx.Button
    _build_board_button: wx.Button
    _reset_button: wx.Button
    _close_camera_button: wx.Button

    # TODO: This will not be hard coded
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

    def __init__(
        self,
        parent: wx.Window,
        connector: Connector,
        status_message_source: StatusMessageSource,
        name: str = "BoardBuilderPanel"
    ):
        super().__init__(
            parent=parent,
            connector=connector,
            status_message_source=status_message_source,
            name=name)

        self._connector = connector
        self._active_request_ids = list()

        ### USER INTERFACE FUNCTIONALITIES AND BUTTONS ###
        horizontal_split_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        control_border_panel: wx.Panel = wx.Panel(parent=self)
        control_border_box: wx.StaticBoxSizer = wx.StaticBoxSizer(
            orient=wx.VERTICAL,
            parent=control_border_panel)
        control_panel: wx.ScrolledWindow = wx.ScrolledWindow(
            parent=control_border_panel)
        control_panel.SetScrollRate(
            xstep=1,
            ystep=1)
        control_panel.ShowScrollbars(
            horz=wx.SHOW_SB_NEVER,
            vert=wx.SHOW_SB_ALWAYS)

        control_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self._board_builder_selector = self.add_control_selector(
            parent=control_panel,
            sizer=control_sizer,
            label="Board Builder",
            selectable_values=list())

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self._reference_marker_id_spinbox = self.add_control_spinbox_integer(
            parent=control_panel,
            sizer=control_sizer,
            label="Reference Marker ID",
            minimum_value=0,
            maximum_value=99,
            initial_value=0)

        self._reference_marker_diameter_spinbox: ParameterSpinboxFloat = self.add_control_spinbox_float(
            parent=control_panel,
            sizer=control_sizer,
            label="Marker diameter (mm)",
            minimum_value=1.0,
            maximum_value=1000.0,
            initial_value=10.0,
            step_value=0.5)

        self._reference_target_submit_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Set Reference Marker")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self._open_camera_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Open Camera"
        )

        self._locate_reference_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Locate Reference"
        )

        self._collect_data_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Collect Data"
        )

        self._build_board_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Build Board"
        )

        self._reset_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Reset"
        )

        self._close_camera_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Close Camera"
        )

        control_spacer_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        control_sizer.Add(
            sizer=control_spacer_sizer,
            flags=wx.SizerFlags(1).Expand())

        control_panel.SetSizerAndFit(sizer=control_sizer)
        control_border_box.Add(
            window=control_panel,
            flags=wx.SizerFlags(1).Expand())

        control_border_panel.SetSizer(sizer=control_border_box)
        horizontal_split_sizer.Add(
            window=control_border_panel,
            flags=wx.SizerFlags(35).Expand())

        self._image_panel = ImagePanel(parent=self)
        self._image_panel.SetBackgroundColour(colour=wx.BLACK)
        horizontal_split_sizer.Add(
            window=self._image_panel,
            flags=wx.SizerFlags(65).Expand())

        self.SetSizerAndFit(sizer=horizontal_split_sizer)


        ### EVENT HANDLING ###
        self._board_builder_selector.selector.Bind(
            event=wx.EVT_CHOICE,
            handler=self.on_board_builder_select)

        self._reference_target_submit_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_reference_target_submit_pressed)

        self._open_camera_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_open_camera_button_click)
        self._locate_reference_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_locate_reference_button_click)
        self._collect_data_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_collect_data_button_click)
        self._build_board_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_build_board_button_click)
        self._reset_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_reset_button_click)
        self._close_camera_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_close_camera_button_click)

        # TODO: IM HERE
        ### INIT ###
        self.cap = None
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_frame, self.timer)
        self._collect_data_button.Enable(False)
        self._build_board_button.Enable(False)
        self._setting_reference = False
        self._collecting_data = False
        self._building_board = False

        ### BOARD BUILDER INIT ###
        self.board_builder = BoardBuilder(self.REFERENCE_MARKER_ID, self.MARKER_SIZE_MM, self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        self.marker_color = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Cyan
        ]

    ### UPDATE ###
    def update_loop(self) -> None:
        super().update_loop()
        self._is_updating = True

        ui_needs_update: bool = False

        if len(self._active_request_ids) > 0:
            completed_request_ids: list[uuid.UUID] = list()
            for request_id in self._active_request_ids:
                _, remaining_request_id = self.update_request(request_id=request_id)
                if remaining_request_id is None:
                    ui_needs_update = True
                    completed_request_ids.append(request_id)
            for request_id in completed_request_ids:
                self._active_request_ids.remove(request_id)
            if len(self._active_request_ids) == 0:
                self.on_active_request_ids_processed()

        if ui_needs_update:
            self._update_controls()

        self._is_updating = False

    def update_frame(self, event):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            wx.MessageBox("Failed to get frame from camera", "Error", wx.OK | wx.ICON_ERROR)
            self.timer.Stop()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners, ids)

        if self._setting_reference:
            self.board_builder.locate_reference_markers(ids, corners)

        elif self._collecting_data:
            if ids is not None:
                corners_dict = self.board_builder.collect_data(ids, corners)
                self.draw_all_corners(corners_dict, frame)

        elif self._building_board:
            if ids is not None:
                corners_dict = self.board_builder.build_board(ids, corners)
                self.draw_all_corners(corners_dict, frame)

        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, frame_rgb)
        self._image_panel.SetBitmap(bitmap)
        self.Refresh()


    def handle_response_series(
        self,
        response_series: MCastResponseSeries,
        task_description: Optional[str] = None,
        expected_response_count: Optional[int] = None
    ) -> bool:
        success: bool = super().handle_response_series(
            response_series=response_series,
            task_description=task_description,
            expected_response_count=expected_response_count)
        if not success:
            return False

        success: bool = True
        response: MCastResponse
        for response in response_series.series:
            if isinstance(response, ErrorResponse):
                self.handle_error_response(response=response)
                success = False
            elif not isinstance(response, EmptyResponse):
                self.handle_unknown_response(response=response)
                success = False
        return success

    def on_active_request_ids_processed(self) -> None:
        if self._current_phase == ACTIVE_PHASE_STARTING_CAPTURE:
            self.status_message_source.enqueue_status_message(
                severity="debug",
                message="ACTIVE_PHASE_STARTING_CAPTURE complete")
            calibrator_labels: list[str] = self._connector.get_connected_calibrator_labels()
            request_series: MCastRequestSeries = MCastRequestSeries(
                series=[ListCalibrationDetectorResolutionsRequest()])
            self._active_request_ids.append(self._connector.request_series_push(
                connection_label=calibrator_labels[0],
                request_series=request_series))
            detector_labels: list[str] = self._connector.get_connected_detector_labels()
            for detector_label in detector_labels:
                request_series: MCastRequestSeries = MCastRequestSeries(
                    series=[GetCapturePropertiesRequest()])
                self._active_request_ids.append(self._connector.request_series_push(
                    connection_label=detector_label,
                    request_series=request_series))
            self._current_phase = ACTIVE_PHASE_STARTING_GET_RESOLUTIONS

    def on_board_builder_select(self, _event: wx.CommandEvent) -> None:
        self._update_controls()

    def on_reference_target_submit_pressed(self, _event: wx.CommandEvent) -> None:
        request_series: MCastRequestSeries = MCastRequestSeries(series=[
            (SetReferenceMarkerRequest(
                marker_id=self._reference_marker_id_spinbox.spinbox.GetValue(),
                marker_diameter=self._reference_marker_diameter_spinbox.spinbox.GetValue()))])
        selected_board_builder_label: str = self._board_builder_selector.selector.GetStringSelection()
        self._active_request_ids.append(self._connector.request_series_push(
            connection_label=selected_board_builder_label,
            request_series=request_series))
        self._update_controls()

    def _update_controls(self) -> None:
        self._board_builder_selector.Enable(False)
        self._reference_marker_id_spinbox.Enable(False)
        self._reference_target_submit_button.Enable(False)
        if len(self._active_request_ids) > 0:
            return  # We're waiting for something
        self._board_builder_selector.Enable(True)
        self._reference_marker_id_spinbox.Enable(True)
        self._reference_target_submit_button.Enable(True)

    ### MAIN BUTTONS ###
    def on_open_camera_button_click(self, event: wx.CommandEvent) -> None:
        # Logic to open the camera goes here
        logger.info("Open Camera button clicked")
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            wx.MessageBox("Cannot open camera", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.timer.Start(1000 // 30)

    def on_locate_reference_button_click(self, event: wx.CommandEvent) -> None:
        logger.info("Set reference button clicked")
        self._collect_data_button.Enable(True)
        self._setting_reference = True
        self._collecting_data = False
        self._building_board = False

    def on_collect_data_button_click(self, event: wx.CommandEvent) -> None:
        logger.info("Collect Data button clicked")
        self._build_board_button.Enable(True)
        self._setting_reference = False
        self._collecting_data = True
        self._building_board = False

    def on_build_board_button_click(self, event: wx.CommandEvent) -> None:
        logger.info("Build Board button clicked")
        self._setting_reference = False
        self._collecting_data = False
        self._building_board = True

    def on_reset_button_click(self, event: wx.CommandEvent) -> None:
        logger.info("Reset button clicked")
        self._collect_data_button.Enable(False)
        self._build_board_button.Enable(False)
        self._setting_reference = False
        self._collecting_data = False
        self._building_board = False
        self.board_builder = BoardBuilder(self.REFERENCE_MARKER_ID, self.MARKER_SIZE_MM, self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)

    def on_close_camera_button_click(self, event: wx.CommandEvent) -> None:
        logger.info("Close Camera button clicked")
        if self.cap is not None and self.cap.isOpened():
            self.timer.Stop()
            self.cap.release()
            self.cap = None
            self._image_panel.SetBitmap(wx.Bitmap())
            self.Refresh()

    def draw_all_corners(self, corners_dict, frame):
        """
        Takes in a dictionary of marker UUIDs to their corners and draws each set of corners on the frame with different colors.
        """
        for index, (marker_uuid, corners_location) in enumerate(corners_dict.items()):
            color_index = index % len(self.marker_color)
            marker_color = self.marker_color[color_index]
            for corner in corners_location:
                x, y, z = corner
                cv2.circle(frame, (int(x) + 200, -int(y) + 200), 4, marker_color, -1)


class ImagePanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.bitmap = None
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def SetBitmap(self, bitmap):
        self.bitmap = bitmap
        self.Refresh()

    def on_paint(self, event):
        dc = wx.PaintDC(self)
        if self.bitmap:
            dc.DrawBitmap(self.bitmap, 0, 0)

