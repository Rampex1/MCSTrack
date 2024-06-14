import cv2
import uuid
import logging
from typing import Final, Optional
import wx
import wx.grid
from cv2 import aruco

from .base_panel import BasePanel
from .feedback import ImagePanel
from .parameters import ParameterSpinboxFloat, ParameterSpinboxInteger

from src.board_builder import BoardBuilder
from src.common.structures import IntrinsicParameters
from src.connector import Connector
from src.common import (
    StatusMessageSource
)

logger = logging.getLogger(__name__)

_UPDATE_INTERVAL_MILLISECONDS: Final[int] = 16

class BoardBuilderPanel(BasePanel):
    _connector: Connector

    _tracked_marker_diameter_spinbox: ParameterSpinboxFloat
    _confirm_marker_size_button: wx.Button
    _open_camera_button: wx.Button
    _close_camera_button: wx.Button
    _locate_reference_button: wx.Button
    _collect_data_button: wx.Button
    _build_board_button: wx.Button

    _image_panel: ImagePanel

    # TODO: User Input
    # TODO: Feature where user chooses origin marker (Right now it just takes the first it reads)
    BOARD_MARKER_IDS = [
            36, 28, 20, 12, 4,
            32, 24, 16, 8, 0,
            37, 29, 21, 13, 5,
            33, 25, 17, 9, 1,
            38, 30, 22, 14, 6,
            34, 26, 18, 10, 2,
            39, 31, 23, 15, 7,
            35, 27, 19, 11, 3
        ]
    MARKER_SIZE_MM: Final[float] = 10.0

    # TODO: This will be determined from calibration
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

        self.cap = None
        self.timer = wx.Timer(self)
        self._setting_reference = False
        self._collecting_data = False
        self._building_board = False

        self.board_builder = BoardBuilder(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        self.marker_size = 0
        self.marker_color = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Cyan
        ]

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

        self._tracked_marker_diameter_spinbox: ParameterSpinboxFloat = self.add_control_spinbox_float(
            parent=control_panel,
            sizer=control_sizer,
            label="Marker diameter (mm)",
            minimum_value=1.0,
            maximum_value=1000.0,
            initial_value=10.0,
            step_value=0.5)

        self._confirm_marker_size_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Confirm marker size")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self.add_text_label(
            parent=control_panel,
            sizer=control_sizer,
            label="Detector",
            font_size_delta=2,
            bold=True)

        self._open_camera_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Open Camera"
        )

        self._close_camera_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Close Camera"
        )

        self.add_text_label(
            parent=control_panel,
            sizer=control_sizer,
            label="Board Builder",
            font_size_delta=2,
            bold=True)

        self._locate_reference_button: wx.ToggleButton = self.add_toggle_button(
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
        self._confirm_marker_size_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_confirm_marker_size_pressed)
        self._open_camera_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_open_camera_button_click)
        self._locate_reference_button.Bind(
            event=wx.EVT_TOGGLEBUTTON,
            handler=self.on_locate_reference_button_click)
        self._collect_data_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_collect_data_button_click)
        self._build_board_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_build_board_button_click)
        self._close_camera_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_close_camera_button_click)

        self._open_camera_button.Enable(False)
        self._close_camera_button.Enable(False)
        self._locate_reference_button.Enable(False)
        self._collect_data_button.Enable(False)
        self._build_board_button.Enable(False)

    ### UPDATE ###
    def _reset(self) -> None:
        logger.info("Reset button clicked")
        self._locate_reference_button.Enable(False)
        self._collect_data_button.Enable(False)
        self._build_board_button.Enable(False)
        self._setting_reference = False
        self._collecting_data = False
        self._building_board = False
        self.board_builder = BoardBuilder(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        self.board_builder.pose_solver.set_marker_size(self._marker_size)

    def update_loop(self) -> None:
        super().update_loop()

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
            self.board_builder.set_intrinsic_parameters(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
            self.board_builder.set_board_marker_ids(self.BOARD_MARKER_IDS)
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
        self._image_panel.set_bitmap(bitmap)
        self.Refresh()

    ### MAIN BUTTONS ###
    def on_confirm_marker_size_pressed(self, _event: wx.CommandEvent) -> None:
        self._marker_size = self._tracked_marker_diameter_spinbox.spinbox.GetValue()
        self.board_builder.pose_solver.set_marker_size(self._marker_size)
        self._open_camera_button.Enable(True)
        self._close_camera_button.Enable(True)

    def on_open_camera_button_click(self, event: wx.CommandEvent) -> None:
        # Logic to open the camera goes here
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            wx.MessageBox("Cannot open camera", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.timer.Start(1000 // 30)
        self._locate_reference_button.Enable(True)

    def on_close_camera_button_click(self, event: wx.CommandEvent) -> None:
        self._reset()
        if self.cap is not None and self.cap.isOpened():
            self.timer.Stop()
            self.cap.release()
            self.cap = None
            self._image_panel.set_bitmap(wx.Bitmap())
            self.Refresh()

    def on_locate_reference_button_click(self, event: wx.CommandEvent) -> None:
        if self._locate_reference_button.GetValue():
            self._reset()
            self._locate_reference_button.SetLabel("Stop Locate Reference")
            self._locate_reference_button.Enable(True)
            self._collect_data_button.Enable(False)
            self._setting_reference = True
            self._collecting_data = False
            self._building_board = False
        else:
            self._locate_reference_button.SetLabel("Locate Reference")
            self._setting_reference = False
            self._collect_data_button.Enable(True)

    def on_collect_data_button_click(self, event: wx.CommandEvent) -> None:
        self._build_board_button.Enable(True)
        self._setting_reference = False
        self._collecting_data = True
        self._building_board = False

    def on_build_board_button_click(self, event: wx.CommandEvent) -> None:
        self._setting_reference = False
        self._collecting_data = False
        self._building_board = True

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
