from .base_panel import BasePanel
from .feedback import ImagePanel
from src.common import StatusMessageSource
from src.connector import Connector
import logging
import wx
import wx.grid

import cv2
from cv2 import aruco
import numpy as np
import datetime
from typing import Final
from src.common.structures import IntrinsicParameters
from src.pose_solver.pose_solver import PoseSolver
from src.pose_solver.pose_solver2 import PoseSolver2
from src.pose_solver.structures import MarkerCorners, TargetMarker, Target
from src.board_builder.board_builder_relative_pose import BoardBuilder


logger = logging.getLogger(__name__)

class BoardBuilderPanel(BasePanel):

    _connector: Connector
    _open_camera_button: wx.Button
    _collect_data_button: wx.Button
    _build_board_button: wx.Button
    _reset_button: wx.Button
    _close_camera_button: wx.Button


    ### CONSTANTS THAT WILL NEED TO BE CHANGED ###
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

        ### BUTTONS ###
        self._open_camera_button = wx.Button(parent=control_panel, label="Open Camera")
        control_sizer.Add(self._open_camera_button, 0, wx.ALL | wx.EXPAND, 5)
        self._open_camera_button.Bind(wx.EVT_BUTTON, self.on_open_camera_button_click)

        self._collect_data_button = wx.Button(parent=control_panel, label="Collect Data")
        control_sizer.Add(self._collect_data_button, 0, wx.ALL | wx.EXPAND, 5)
        self._collect_data_button.Bind(wx.EVT_BUTTON, self.on_collect_data_button_click)

        self._build_board_button = wx.Button(parent=control_panel, label="Build Board")
        control_sizer.Add(self._build_board_button, 0, wx.ALL | wx.EXPAND, 5)
        self._build_board_button.Bind(wx.EVT_BUTTON, self.on_build_board_button_click)

        self._reset_button = wx.Button(parent=control_panel, label="Reset")
        control_sizer.Add(self._reset_button, 0, wx.ALL | wx.EXPAND, 5)
        self._reset_button.Bind(wx.EVT_BUTTON, self.on_reset_button_click)

        self._close_camera_button = wx.Button(parent=control_panel, label="Close Camera")
        control_sizer.Add(self._close_camera_button, 0, wx.ALL | wx.EXPAND, 5)
        self._close_camera_button.Bind(wx.EVT_BUTTON, self.on_close_camera_button_click)

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

        self.cap = None
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_frame, self.timer)
        self.collecting_data = False
        self.building_board = False

        ### CAMERA SETTINGS ###
        self.camera_matrix = np.array([
            [self.DETECTOR_GREEN_INTRINSICS.focal_length_x_px, 0, self.DETECTOR_GREEN_INTRINSICS.optical_center_x_px],
            [0, self.DETECTOR_GREEN_INTRINSICS.focal_length_y_px, self.DETECTOR_GREEN_INTRINSICS.optical_center_y_px],
            [0, 0, 1]], dtype=float)
        self.dist_coeffs = np.array(self.DETECTOR_GREEN_INTRINSICS.radial_distortion_coefficients
                                    + self.DETECTOR_GREEN_INTRINSICS.tangential_distortion_coefficients)

        ### POSE SOLVER INIT ###
        self.pose_solver = PoseSolver()
        self.pose_solver.set_intrinsic_parameters(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        self.pose_solver.set_reference_target(TargetMarker(
            marker_id=self.REFERENCE_MARKER_ID,
            marker_size=self.MARKER_SIZE_MM))
        self.pose_solver2 = PoseSolver2()
        self.pose_solver2.set_intrinsic_parameters(self.DETECTOR_GREEN_NAME, self.DETECTOR_GREEN_INTRINSICS)
        self._target_poses = []
        self._detector_poses = []
        self._targets = []

        ### BOARD BUILDER INIT ###
        self.board_builder = BoardBuilder()
        self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS = 20
        self.local_corners = np.array([
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-left
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Top-right
            [self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1],  # Bottom-right
            [-self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, -self.MARKER_MIDDLE_TO_EDGE_IN_PIXELS, 0, 1]  # Bottom-left
        ])
        self.marker_color = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Cyan
        ]

    ### UPDATE ###

    def update_loop(self) -> None:
        super().update_loop()

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

        if self.collecting_data:
            self.solve_pose(ids, corners)
            if self._detector_poses is not None:
                self.board_builder.add_detector_poses(self._detector_poses)

            if self._target_poses is not None:
                self.board_builder.add_target_poses(self._target_poses)
                self.board_builder.collect_data()

                ### DRAW MARKERS ###
                for index, pose in enumerate(self._target_poses):
                    pose_values = pose.object_to_reference_matrix.values
                    pose_matrix = np.array(pose_values).reshape(4, 4)
                    color_index = index % len(self.marker_color)
                    corners_location = self.calculate_corners_location(pose_matrix, self.local_corners)
                    self.draw_corners_location(corners_location, frame, self.marker_color[color_index])

        elif self.building_board:
            self.solve_pose2(ids, corners)

            if self._target_poses is not None:
                self.board_builder.add_target_poses(self._target_poses)
                corners_dict = self.board_builder.build_board(self.pose_solver2)
                for index, marker_uuid in enumerate(corners_dict):
                    color_index = index % len(self.marker_color)
                    self.draw_corners_location(corners_dict[marker_uuid], frame, self.marker_color[color_index])


        height, width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bitmap = wx.Bitmap.FromBuffer(width, height, frame_rgb)
        self._image_panel.SetBitmap(bitmap)
        self.Refresh()

    def on_open_camera_button_click(self, event: wx.CommandEvent) -> None:
        # Logic to open the camera goes here
        logger.info("Open Camera button clicked")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            wx.MessageBox("Cannot open camera", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.timer.Start(1000//30)

    def on_collect_data_button_click(self, event: wx.CommandEvent) -> None:
        logger.info("Collect Data button clicked")
        self.collecting_data = True
        self.building_board = False

    def on_build_board_button_click(self, event: wx.CommandEvent) -> None:
        self.collecting_data = False
        self.building_board = True

    def on_reset_button_click(self, event: wx.CommandEvent) -> None:
        self.collecting_data = False
        self.building_board = False
        self._detector_poses = []
        self._target_poses = []
        self.board_builder = BoardBuilder()

    def on_close_camera_button_click(self, event: wx.CommandEvent) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.timer.Stop()
            self.cap.release()
            self.cap = None
            self._image_panel.SetBitmap(wx.Bitmap())  # Clear the image panel
            self.Refresh()

    def solve_pose(self, ids, corners):
        """ Given visible Ids and their corners, uses pose_solver to return the pose in the reference frame """

        if ids is not None:
            ### ADD TARGET MARKER ###
            for marker_id in range(len(ids)):
                if ids[marker_id][0] != self.REFERENCE_MARKER_ID:
                    if self.pose_solver.try_add_target_marker(ids[marker_id][0], int(self.MARKER_SIZE_MM)):
                        self.board_builder.expand_matrix()


            ### ADD CORNERS ###
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label=self.DETECTOR_GREEN_NAME,
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),
                    timestamp=datetime.datetime.now()
                )
                self.pose_solver.add_marker_corners([marker_corners])
                self.pose_solver2.add_marker_corners([marker_corners])

            ### SOLVE POSE ###
            self.pose_solver.update()
            detector_poses, target_poses = self.pose_solver.get_poses()
            # {'default_camera': Matrix4x4(values=[-0.026744524016976357, 0.9314369559288025, -0.3629186153411865, -47.215579986572266, -0.4845777153968811, 0.30546122789382935, 0.8196815252304077, 139.50146484375, 0.8743392825126648, 0.19778425991535187, 0.44318416714668274, 192.38699340820312, 0.0, 0.0, 0.0, 1.0])}
            self._detector_poses = detector_poses
            self._target_poses = target_poses

    def solve_pose2(self, ids, corners):
        """ Given visible Ids and their corners, uses pose_solver to return the pose in the reference frame """

        self._targets = self.pose_solver.get_targets()
        self.pose_solver2.add_target_marker(self._targets)

        if ids is not None:
            ### ADD CORNERS ###
            for i, corner in enumerate(corners):
                marker_corners = MarkerCorners(
                    detector_label=self.DETECTOR_GREEN_NAME,
                    marker_id=int(ids[i][0]),
                    points=corner[0].tolist(),
                    timestamp=datetime.datetime.now()
                )
                self.pose_solver2.add_marker_corners([marker_corners])

            ### SOLVE POSE ###
            _, target_poses = self.pose_solver2.get_poses()
            self._target_poses = target_poses

    def calculate_corners_location(self, T_matrix, local_corners):
        """ Given a matrix transformation, find the four corners """
        corners_reference = np.zeros((4, 4))
        for i in range(4):
            corners_reference[i] = T_matrix @ local_corners[i]

        corners_reference = corners_reference[:, :3]
        return corners_reference

    def draw_corners_location(self, corners_location, frame, marker_color):
        """ Takes in a list of three coordinates (x,y,z) and draws it on the board with different colors every 4 iterations."""

        for corner in corners_location:
            x, y, z = corner
            cv2.circle(frame, (int(x) + 200, - int(y) + 200), 4, marker_color, -1)


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
