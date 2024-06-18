from .base_panel import \
    BasePanel
from .parameters import \
    ParameterSelector, \
    ParameterSpinboxFloat, \
    ParameterSpinboxInteger
from .specialized import \
    GraphicsRenderer, \
    TrackingTable, \
    TrackingTableRow
from src.common import \
    ErrorResponse, \
    EmptyResponse, \
    MCastRequestSeries, \
    MCastResponse, \
    MCastResponseSeries, \
    StatusMessageSource
from src.common.structures import \
    DetectorFrame, \
    Matrix4x4, \
    Pose, \
    PoseSolverFrame
from src.connector import \
    Connector
from src.pose_solver.api import \
    AddTargetMarkerRequest, \
    AddTargetMarkerResponse, \
    SetReferenceMarkerRequest
import datetime
import logging
import platform
from typing import Final, Optional
import uuid
import wx
import wx.grid


logger = logging.getLogger(__name__)

POSE_REPRESENTATIVE_MODEL: Final[str] = "coordinate_axes"


class PoseSolverPanel(BasePanel):

    _connector: Connector

    _pose_solver_selector: ParameterSelector
    _reference_marker_id_spinbox: ParameterSpinboxInteger
    _reference_marker_diameter_spinbox: ParameterSpinboxFloat
    _reference_target_submit_button: wx.Button
    _tracked_marker_id_spinbox: ParameterSpinboxInteger
    _tracked_marker_diameter_spinbox: ParameterSpinboxFloat
    _tracked_target_submit_button: wx.Button
    _tracking_start_button: wx.Button
    _tracking_stop_button: wx.Button
    _tracking_table: TrackingTable

    _active_request_ids: list[uuid.UUID]
    _is_updating: bool
    _is_waiting_for_connector: bool
    _latest_detector_frames: dict[str, DetectorFrame]  # last frame for each detector
    _latest_pose_solver_frames: dict[str, PoseSolverFrame]
    _target_id_to_label: dict[str, str]
    _tracked_target_poses: list[Pose]

    def __init__(
        self,
        parent: wx.Window,
        connector: Connector,
        status_message_source: StatusMessageSource,
        name: str = "PoseSolverPanel"
    ):
        super().__init__(
            parent=parent,
            connector=connector,
            status_message_source=status_message_source,
            name=name)
        self._connector = connector

        self._active_request_ids = list()
        self._is_updating = False
        self._is_waiting_for_connector = False
        self._latest_detector_frames = dict()
        self._latest_pose_solver_frames = dict()
        self._target_id_to_label = dict()
        self._tracked_target_poses = list()

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

        self._pose_solver_selector = self.add_control_selector(
            parent=control_panel,
            sizer=control_sizer,
            label="Pose Solver",
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

        self._tracked_marker_id_spinbox = self.add_control_spinbox_integer(
            parent=control_panel,
            sizer=control_sizer,
            label="Tracked Marker ID",
            minimum_value=0,
            maximum_value=99,
            initial_value=1)

        self._tracked_marker_diameter_spinbox: ParameterSpinboxFloat = self.add_control_spinbox_float(
            parent=control_panel,
            sizer=control_sizer,
            label="Marker diameter (mm)",
            minimum_value=1.0,
            maximum_value=1000.0,
            initial_value=10.0,
            step_value=0.5)

        self._tracked_target_submit_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Add Tracked Marker")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self._tracking_start_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Start Tracking")

        self._tracking_stop_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Stop Tracking")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self._tracking_table = TrackingTable(parent=control_panel)
        control_sizer.Add(
            window=self._tracking_table,
            flags=wx.SizerFlags(0).Expand())
        control_sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)

        self._tracking_display_textbox = wx.TextCtrl(
            parent=control_panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH)
        self._tracking_display_textbox.SetEditable(False)
        self._tracking_display_textbox.SetBackgroundColour(colour=wx.Colour(red=249, green=249, blue=249, alpha=255))
        control_sizer.Add(
            window=self._tracking_display_textbox,
            flags=wx.SizerFlags(1).Align(wx.EXPAND))

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

        if platform.system() == "Linux":
            logger.warning("OpenGL context creation does not currently work well in Linux. Rendering is disabled.")
            self._renderer = None
        else:
            self._renderer = GraphicsRenderer(parent=self)
            horizontal_split_sizer.Add(
                window=self._renderer,
                flags=wx.SizerFlags(65).Expand())
            self._renderer.load_models_into_context_from_data_path()
            self._renderer.add_scene_object("coordinate_axes", Matrix4x4())

        self.SetSizerAndFit(sizer=horizontal_split_sizer)

        self._pose_solver_selector.selector.Bind(
            event=wx.EVT_CHOICE,
            handler=self.on_pose_solver_select)
        self._reference_target_submit_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_reference_target_submit_pressed)
        self._tracked_target_submit_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_tracked_target_submit_pressed)
        self._tracking_start_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_tracking_start_pressed)
        self._tracking_stop_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_tracking_stop_pressed)
        self._tracking_table.table.Bind(
            event=wx.grid.EVT_GRID_SELECT_CELL,
            handler=self.on_tracking_row_selected)

    def handle_error_response(
        self,
        response: ErrorResponse
    ):
        super().handle_error_response(response=response)

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
            if isinstance(response, AddTargetMarkerResponse):
                success = True  # we don't currently do anything with this response in this interface
            elif isinstance(response, ErrorResponse):
                self.handle_error_response(response=response)
                success = False
            elif not isinstance(response, EmptyResponse):
                self.handle_unknown_response(response=response)
                success = False
        return success

    def on_page_select(self) -> None:
        super().on_page_select()
        selected_pose_solver_label: str = self._pose_solver_selector.selector.GetStringSelection()
        available_pose_solver_labels: list[str] = self._connector.get_connected_pose_solver_labels()
        self._pose_solver_selector.set_options(option_list=available_pose_solver_labels)
        if selected_pose_solver_label in available_pose_solver_labels:
            self._pose_solver_selector.selector.SetStringSelection(selected_pose_solver_label)
        else:
            self._pose_solver_selector.selector.SetStringSelection(str())
        self._update_controls()

    def on_pose_solver_select(self, _event: wx.CommandEvent) -> None:
        self._update_controls()

    def on_reference_target_submit_pressed(self, _event: wx.CommandEvent) -> None:
        request_series: MCastRequestSeries = MCastRequestSeries(series=[
            (SetReferenceMarkerRequest(
                marker_id=self._reference_marker_id_spinbox.spinbox.GetValue(),
                marker_diameter=self._reference_marker_diameter_spinbox.spinbox.GetValue()))])
        selected_pose_solver_label: str = self._pose_solver_selector.selector.GetStringSelection()
        self._active_request_ids.append(self._connector.request_series_push(
            connection_label=selected_pose_solver_label,
            request_series=request_series))
        self._update_controls()

    def on_tracked_target_submit_pressed(self, _event: wx.CommandEvent) -> None:
        request_series: MCastRequestSeries = MCastRequestSeries(series=[
            (AddTargetMarkerRequest(
                marker_id=self._tracked_marker_id_spinbox.spinbox.GetValue(),
                marker_diameter=self._tracked_marker_diameter_spinbox.spinbox.GetValue()))])
        selected_pose_solver_label: str = self._pose_solver_selector.selector.GetStringSelection()
        self._active_request_ids.append(self._connector.request_series_push(
            connection_label=selected_pose_solver_label,
            request_series=request_series))
        self._update_controls()

    def on_tracking_row_selected(self, _event: wx.grid.GridEvent) -> None:
        if self._is_updating:
            return
        selected_index: int | None = self._tracking_table.get_selected_row_index()
        if selected_index is not None:
            if 0 <= selected_index < len(self._tracked_target_poses):
                display_text: str = self._tracked_target_poses[selected_index].json(indent=4)
                self._tracking_display_textbox.SetValue(display_text)
            else:
                self.status_message_source.enqueue_status_message(
                    severity="error",
                    message=f"Target index {selected_index} is out of bounds. Selection will be set to None.")
                self._tracking_table.set_selected_row_index(None)
        self._update_controls()

    def on_tracking_start_pressed(self, _event: wx.CommandEvent) -> None:
        self._connector.start_tracking(mode=Connector.StartupMode.DETECTING_AND_SOLVING)
        self._is_waiting_for_connector = True
        self._update_controls()

    def on_tracking_stop_pressed(self, _event: wx.CommandEvent) -> None:
        self._tracking_table.update_contents(list())
        self._tracking_table.Enable(False)
        self._tracking_display_textbox.SetValue(str())
        self._tracking_display_textbox.Enable(False)
        self._connector.stop_tracking()
        self._is_waiting_for_connector = True
        self._update_controls()

    def update_loop(self) -> None:
        super().update_loop()

        if self._renderer is not None:
            self._renderer.render()

        self._is_updating = True

        ui_needs_update: bool = False
        if self._is_waiting_for_connector:
            if not self._connector.is_in_transition():
                ui_needs_update = True
                self._is_waiting_for_connector = False

        if self._connector.is_running():
            detector_labels: list[str] = self._connector.get_connected_detector_labels()
            for detector_label in detector_labels:
                retrieved_detector_frame: DetectorFrame = self._connector.get_live_detector_frame(
                    detector_label=detector_label)
                retrieved_detector_frame_timestamp: datetime.datetime = retrieved_detector_frame.timestamp_utc()
                if detector_label in self._latest_detector_frames:
                    latest_detector_frame: DetectorFrame = self._latest_detector_frames[detector_label]
                    latest_detector_frame_timestamp: datetime.datetime = latest_detector_frame.timestamp_utc()
                    if retrieved_detector_frame_timestamp > latest_detector_frame_timestamp:
                        self._latest_detector_frames[detector_label] = retrieved_detector_frame
                else:
                    self._latest_detector_frames[detector_label] = retrieved_detector_frame

            new_poses_available: bool = False
            pose_solver_labels: list[str] = self._connector.get_connected_pose_solver_labels()
            for pose_solver_label in pose_solver_labels:
                retrieved_pose_solver_frame: PoseSolverFrame = self._connector.get_live_pose_solver_frame(
                    pose_solver_label=pose_solver_label)
                retrieved_pose_solver_frame_timestamp: datetime.datetime = retrieved_pose_solver_frame.timestamp_utc()
                if pose_solver_label in self._latest_pose_solver_frames:
                    latest_pose_solver_frame: PoseSolverFrame = self._latest_pose_solver_frames[pose_solver_label]
                    latest_pose_solver_frame_timestamp: datetime.datetime = latest_pose_solver_frame.timestamp_utc()
                    if retrieved_pose_solver_frame_timestamp > latest_pose_solver_frame_timestamp:
                        self._latest_pose_solver_frames[pose_solver_label] = retrieved_pose_solver_frame
                        new_poses_available = True
                else:
                    self._latest_pose_solver_frames[pose_solver_label] = retrieved_pose_solver_frame
                    new_poses_available = True
            if new_poses_available:
                self._tracked_target_poses.clear()
                if self._renderer is not None:
                    self._renderer.clear_scene_objects()
                    self._renderer.add_scene_object(  # Reference
                        model_key=POSE_REPRESENTATIVE_MODEL,
                        transform_to_world=Matrix4x4())
                table_rows: list[TrackingTableRow] = list()
                for live_pose_solver in self._latest_pose_solver_frames.values():
                    for pose in live_pose_solver.target_poses:
                        label: str = str()
                        if pose.target_id in self._target_id_to_label:
                            label = self._target_id_to_label[pose.target_id]
                        table_row: TrackingTableRow = TrackingTableRow(
                            target_id=pose.target_id,
                            label=label,
                            x=pose.object_to_reference_matrix[0, 3],
                            y=pose.object_to_reference_matrix[1, 3],
                            z=pose.object_to_reference_matrix[2, 3])
                        table_rows.append(table_row)
                        self._tracked_target_poses.append(pose)
                        if self._renderer is not None:
                            self._renderer.add_scene_object(
                                model_key=POSE_REPRESENTATIVE_MODEL,
                                transform_to_world=pose.object_to_reference_matrix)
                    for pose in live_pose_solver.detector_poses:
                        table_row: TrackingTableRow = TrackingTableRow(
                            target_id=pose.target_id,
                            label=pose.target_id,
                            x=pose.object_to_reference_matrix[0, 3],
                            y=pose.object_to_reference_matrix[1, 3],
                            z=pose.object_to_reference_matrix[2, 3])
                        table_rows.append(table_row)
                        self._tracked_target_poses.append(pose)
                        if self._renderer is not None:
                            self._renderer.add_scene_object(
                                model_key=POSE_REPRESENTATIVE_MODEL,
                                transform_to_world=pose.object_to_reference_matrix)
                self._tracking_table.update_contents(row_contents=table_rows)
                if len(table_rows) > 0:
                    self._tracking_table.Enable(True)
                else:
                    self._tracking_table.Enable(False)

        if len(self._active_request_ids) > 0:
            completed_request_ids: list[uuid.UUID] = list()
            for request_id in self._active_request_ids:
                _, remaining_request_id = self.update_request(request_id=request_id)
                if remaining_request_id is None:
                    ui_needs_update = True
                    completed_request_ids.append(request_id)
            for request_id in completed_request_ids:
                self._active_request_ids.remove(request_id)

        if ui_needs_update:
            self._update_controls()

        self._is_updating = False

    def _update_controls(self) -> None:
        self._pose_solver_selector.Enable(False)
        self._reference_marker_id_spinbox.Enable(False)
        self._reference_target_submit_button.Enable(False)
        self._tracked_marker_id_spinbox.Enable(False)
        self._tracked_target_submit_button.Enable(False)
        self._tracking_start_button.Enable(False)
        self._tracking_stop_button.Enable(False)
        self._tracking_table.Enable(False)
        self._tracking_display_textbox.Enable(False)
        if len(self._active_request_ids) > 0 or self._connector.is_in_transition():
            return  # We're waiting for something
        self._pose_solver_selector.Enable(True)
        self._reference_marker_id_spinbox.Enable(True)
        self._reference_target_submit_button.Enable(True)
        self._tracked_marker_id_spinbox.Enable(True)
        self._tracked_target_submit_button.Enable(True)
        if not self._connector.is_running():
            self._tracking_start_button.Enable(True)
        else:
            self._tracking_stop_button.Enable(True)
        if len(self._tracked_target_poses) > 0:
            self._tracking_table.Enable(True)
            tracked_target_index: int = self._tracking_table.get_selected_row_index()
            if tracked_target_index is not None:
                if tracked_target_index >= len(self._tracked_target_poses):
                    self.status_message_source.enqueue_status_message(
                        severity="warning",
                        message=f"Selected tracked target index {tracked_target_index} is out of bounds. "
                                "Setting to None.")
                    self._tracking_table.set_selected_row_index(None)
                else:
                    self._tracking_display_textbox.Enable(True)
