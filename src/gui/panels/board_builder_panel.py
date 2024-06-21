import cv2
import logging
from typing import Final
import wx
import wx.grid
from cv2 import aruco
import datetime

from .base_panel import BasePanel
from .feedback import ImagePanel
from .parameters import ParameterSpinboxFloat

from src.board_builder import BoardBuilder
from src.common.structures import IntrinsicParameters, PoseSolverFrame, Pose, Matrix4x4
from src.connector import Connector
from src.common import (
    StatusMessageSource
)
from .pose_solver_panel import POSE_REPRESENTATIVE_MODEL
from .specialized import \
    GraphicsRenderer

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
    _tracked_target_poses: list[Pose]
    _latest_pose_solver_frames: dict[str, PoseSolverFrame]

    # TODO: This will be determined by calibration
    DETECTOR_GREEN_NAME: Final[str] = "detector_green"
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

    DETECTOR_BLUE_NAME: Final[str] = "detector_blue"
    DETECTOR_BLUE_INTRINSICS: Final[IntrinsicParameters] = IntrinsicParameters(
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
        self.cap1 = None  # Second camera capture
        self.timer = wx.Timer(self)
        self._locating_reference = False
        self._collecting_data = False
        self._building_board = False

        self._tracked_target_poses = list()
        self._latest_pose_solver_frames = dict()
        self._detector_data = {
            self.DETECTOR_GREEN_NAME: {
                "ids": None,
                "corners": None,
                "intrinsics": self.DETECTOR_GREEN_INTRINSICS
            },
            self.DETECTOR_BLUE_NAME: {
                "ids": None,
                "corners": None,
                "intrinsics": self.DETECTOR_BLUE_INTRINSICS
            }
        }

        self.board_builder = BoardBuilder()
        self._marker_size = 0
        self.marker_color = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Cyan
        ]

        ### USER INTERFACE FUNCTIONALITIES AND BUTTONS ###
        horizontal_split_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)  # Changed from HORIZONTAL to VERTICAL

        control_border_panel: wx.Panel = wx.Panel(parent=self)
        control_border_box: wx.StaticBoxSizer = wx.StaticBoxSizer(
            wx.VERTICAL,
            control_border_panel,
            label="Control Panel")
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

        self._collect_data_button: wx.ToggleButton = self.add_toggle_button(
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
            control_panel,
            proportion=1,
            flag=wx.EXPAND)

        control_border_panel.SetSizer(control_border_box)
        horizontal_split_sizer.Add(
            control_border_panel,
            proportion=1,
            flag=wx.EXPAND)

        camera_split_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)  # New sizer for two camera frames
        self._image_panel0 = ImagePanel(parent=self)  # Top camera frame (Camera 0)
        self._image_panel0.SetBackgroundColour(colour=wx.BLACK)
        camera_split_sizer.Add(self._image_panel0, proportion=1, flag=wx.EXPAND)

        self._image_panel1 = ImagePanel(parent=self)  # Bottom camera frame (Camera 1)
        self._image_panel1.SetBackgroundColour(colour=wx.BLACK)
        camera_split_sizer.Add(self._image_panel1, proportion=1, flag=wx.EXPAND)

        horizontal_split_sizer.Add(
            camera_split_sizer,
            proportion=1,
            flag=wx.EXPAND)  # Added new sizer to the main sizer

        self._renderer = GraphicsRenderer(parent=self)
        horizontal_split_sizer.Add(
            self._renderer,
            proportion=1,
            flag=wx.EXPAND)  # Adjusted flag value to balance the new layout
        self._renderer.load_models_into_context_from_data_path()
        self._renderer.add_scene_object("coordinate_axes", Matrix4x4())

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
            event=wx.EVT_TOGGLEBUTTON,
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
        self._locating_reference = False
        self._collecting_data = False
        self._building_board = False
        self.board_builder = BoardBuilder()

    def _render_frame(self, detector_poses, target_poses):
        pose_solver_frame = PoseSolverFrame(
            detector_poses=detector_poses,
            target_poses=target_poses,
            timestamp_utc_iso8601=str(datetime.datetime.now())
        )

        ### RENDERER ###
        self._tracked_target_poses.clear()
        if self._renderer is not None:
            self._latest_pose_solver_frames['pose_solver_label'] = pose_solver_frame
            self._renderer.clear_scene_objects()
            self._renderer.add_scene_object(  # Reference
                model_key=POSE_REPRESENTATIVE_MODEL,
                transform_to_world=Matrix4x4())
        for live_pose_solver in self._latest_pose_solver_frames.values():
            for pose in live_pose_solver.target_poses:
                self._tracked_target_poses.append(pose)
                if self._renderer is not None:
                    self._renderer.add_scene_object(
                        model_key=POSE_REPRESENTATIVE_MODEL,
                        transform_to_world=pose.object_to_reference_matrix)
                if self._renderer is not None:
                    self._renderer.add_scene_object(
                        model_key=POSE_REPRESENTATIVE_MODEL,
                        transform_to_world=pose.object_to_reference_matrix)


    def update_loop(self) -> None:
        # Existing super call
        super().update_loop()

        # Capture frames from both cameras
        if (self.cap is None or not self.cap.isOpened()) or (self.cap1 is None or not self.cap1.isOpened()):
            return

        if self._renderer is not None:
            self._renderer.render()

        ret0, frame0 = self.cap.read()
        ret1, frame1 = self.cap1.read()

        if not ret0 or not ret1:
            wx.MessageBox("Failed to get frame from one or both cameras", "Error", wx.OK | wx.ICON_ERROR)
            self.timer.Stop()
            return

        self.process_frame(frame0, self.DETECTOR_GREEN_NAME, self._image_panel0)
        self.process_frame(frame1, self.DETECTOR_BLUE_NAME, self._image_panel1)

        self.Refresh()

    def process_frame(self, frame, detector_name, image_panel):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is not None and corners is not None:
            reformatted_ids = [single_id[0] for single_id in ids]
            reformatted_corners = [corner[0] for corner in corners]
            self._detector_data[detector_name]['ids'] = reformatted_ids
            self._detector_data[detector_name]['corners'] = reformatted_corners

        if self._locating_reference:
            for detector_name in self._detector_data:
                self.board_builder.pose_solver.set_intrinsic_parameters(detector_name, self._detector_data[detector_name]["intrinsics"])
            self.board_builder.locate_reference_markers(self._detector_data)

        elif self._collecting_data:
            if (self._detector_data[self.DETECTOR_GREEN_NAME]['ids'] is not None or
                    self._detector_data[self.DETECTOR_BLUE_NAME]['ids'] is not None):
                corners_dict = self.board_builder.collect_data(self._detector_data)
                # TODO: We want to draw different markers for each frame
                self.draw_all_corners(corners_dict, frame)
                self._render_frame(self.board_builder.detector_poses, self.board_builder.target_poses)

        elif self._building_board:
            if (self._detector_data[self.DETECTOR_GREEN_NAME]['ids'] is not None or
                    self._detector_data[self.DETECTOR_BLUE_NAME]['ids'] is not None):
                corners_dict = self.board_builder.build_board(self._detector_data)
                # TODO: We want to draw different markers for each frame
                self.draw_all_corners(corners_dict, frame)
                self._render_frame(self.board_builder.detector_poses,
                                               self.board_builder.target_poses + self.board_builder.occluded_poses)

        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, frame_rgb)
        image_panel.set_bitmap(bitmap)

    ### MAIN BUTTONS ###
    def on_confirm_marker_size_pressed(self, _event: wx.CommandEvent) -> None:
        self._marker_size = self._tracked_marker_diameter_spinbox.spinbox.GetValue()
        self.board_builder.pose_solver.set_board_marker_size(self._marker_size)
        self._open_camera_button.Enable(True)
        self._close_camera_button.Enable(True)

    def on_open_camera_button_click(self, event: wx.CommandEvent) -> None:
        self.cap = cv2.VideoCapture(1)
        self.cap1 = cv2.VideoCapture(2)
        if not self.cap.isOpened() or not self.cap1.isOpened():
            wx.MessageBox("Cannot open one or both cameras", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.timer.Start(1000 // 30)
        self._locate_reference_button.Enable(True)

    def on_close_camera_button_click(self, event: wx.CommandEvent) -> None:
        self._reset()
        if (self.cap is not None and self.cap.isOpened()) or (self.cap1 is not None and self.cap1.isOpened()):
            self.timer.Stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.cap1 is not None:
                self.cap1.release()
                self.cap1 = None
            self._image_panel0.set_bitmap(wx.Bitmap(1, 1))
            self._image_panel1.set_bitmap(wx.Bitmap(1, 1))
            self.Refresh()

    def on_locate_reference_button_click(self, event: wx.CommandEvent) -> None:
        if self._locate_reference_button.GetValue():
            self._locate_reference_button.SetLabel("Stop Locate Reference")
            self._locate_reference_button.Enable(True)
            self._collect_data_button.Enable(False)
            self._locating_reference = True
            self._collecting_data = False
            self._building_board = False
        else:
            self._locate_reference_button.SetLabel("Locate Reference")
            self._locating_reference = False
            self._collect_data_button.Enable(True)

    def on_collect_data_button_click(self, event: wx.CommandEvent) -> None:
        if self._collect_data_button.GetValue():
            self._collect_data_button.SetLabel("Stop Collect Data")
            self._build_board_button.Enable(True)
            self._locating_reference = False
            self._collecting_data = True
            self._building_board = False
        else:
            self._collect_data_button.SetLabel("Collect Data")
            self._collecting_data = False

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
                cv2.circle(frame, (int(x) + 300, -int(y) + 300), 4, marker_color, -1)
