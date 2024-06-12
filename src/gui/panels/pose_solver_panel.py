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
from src.calibrator.api import \
    GetCalibrationResultRequest, \
    GetCalibrationResultResponse, \
    ListCalibrationDetectorResolutionsRequest, \
    ListCalibrationDetectorResolutionsResponse, \
    ListCalibrationResultMetadataRequest, \
    ListCalibrationResultMetadataResponse
from src.common import \
    ErrorResponse, \
    EmptyResponse, \
    MCastRequest, \
    MCastRequestSeries, \
    MCastResponse, \
    MCastResponseSeries, \
    StatusMessageSource
from src.common.structures import \
    DetectorResolution, \
    ImageResolution, \
    IntrinsicParameters, \
    MarkerSnapshot, \
    Matrix4x4, \
    Pose
from src.connector import \
    Connector
from src.detector.api import \
    GetCapturePropertiesRequest, \
    GetCapturePropertiesResponse, \
    GetMarkerSnapshotsRequest, \
    GetMarkerSnapshotsResponse, \
    StartCaptureRequest, \
    StopCaptureRequest
from src.pose_solver.api import \
    AddTargetMarkerRequest, \
    AddTargetMarkerResponse, \
    AddMarkerCornersRequest, \
    GetPosesRequest, \
    GetPosesResponse, \
    SetIntrinsicParametersRequest, \
    SetReferenceMarkerRequest, \
    StartPoseSolverRequest, \
    StopPoseSolverRequest
import datetime
import logging
from typing import Final, Optional
import uuid
import wx
import wx.grid


logger = logging.getLogger(__name__)


# Active means that something is happening that should prevent the user from interacting with the UI
ACTIVE_PHASE_IDLE: Final[int] = 0
ACTIVE_PHASE_STARTING_CAPTURE: Final[int] = 1
ACTIVE_PHASE_STARTING_GET_RESOLUTIONS: Final[int] = 2
ACTIVE_PHASE_STARTING_LIST_INTRINSICS: Final[int] = 3  # This and next phase to be combined with modified API
ACTIVE_PHASE_STARTING_GET_INTRINSICS: Final[int] = 4
ACTIVE_PHASE_STARTING_FINAL: Final[int] = 5
ACTIVE_PHASE_STOPPING: Final[int] = 6

POSE_REPRESENTATIVE_MODEL: Final[str] = "coordinate_axes"


# TODO: There is a lot of general logic that probably best be moved to the Connector class,
#       thereby allowing the pose solving functionality to be used without the UI.
class PoseSolverPanel(BasePanel):

    _connector: Connector

    _active_request_ids: list[uuid.UUID]

    class PassiveDetectorRequest:
        request_id: uuid.UUID | None
        detected_marker_snapshots: list[MarkerSnapshot]
        rejected_marker_snapshots: list[MarkerSnapshot]
        marker_snapshot_timestamp: datetime.datetime
        def __init__(self):
            self.request_id = None
            self.detected_marker_snapshots = list()
            self.rejected_marker_snapshots = list()
            self.marker_snapshot_timestamp = datetime.datetime.min
    _passive_detecting_requests: dict[str, PassiveDetectorRequest]  # access by detector_label
    _passive_solving_request_id: uuid.UUID | None

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

    _is_solving: bool
    _is_updating: bool
    _current_phase: int
    _target_id_to_label: dict[str, str]
    _tracked_target_poses: list[Pose]

    # Variables assigned upon starting the pose solver
    _detector_calibration_labels: dict[str, str]
    _detector_resolutions: dict[str, ImageResolution]
    _calibrated_resolutions: list[DetectorResolution]
    _detector_intrinsics: dict[str, IntrinsicParameters]

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
        self._passive_detecting_requests = dict()
        self._passive_solving_request_id = None

        self._is_solving = False
        self._is_updating = False
        self._current_phase = ACTIVE_PHASE_IDLE
        self._tracked_target_poses = list()
        self._target_id_to_label = dict()

        self._detector_resolutions = dict()
        self._calibrated_resolutions = list()
        self._detector_calibration_labels = dict()
        self._detector_intrinsics = dict()

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

        self._renderer = GraphicsRenderer(parent=self)
        self._renderer.load_models_into_context_from_data_path()
        self._renderer.add_scene_object("coordinate_axes", Matrix4x4())
        horizontal_split_sizer.Add(
            window=self._renderer,
            flags=wx.SizerFlags(65).Expand())

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

    def handle_response_get_calibration_result(
        self,
        response: GetCalibrationResultResponse
    ) -> None:
        self._detector_intrinsics[response.intrinsic_calibration.detector_serial_identifier] = \
            response.intrinsic_calibration.calibrated_values

    def handle_response_get_capture_properties(
        self,
        response: GetCapturePropertiesResponse,
        detector_label: str
    ) -> None:
        self._detector_resolutions[detector_label] = ImageResolution(
            x_px=response.resolution_x_px,
            y_px=response.resolution_y_px)

    def handle_response_get_marker_snapshots(
        self,
        response: GetMarkerSnapshotsResponse,
        detector_label: str
    ):
        if detector_label in self._passive_detecting_requests.keys():
            self._passive_detecting_requests[detector_label].detected_marker_snapshots = \
                response.detected_marker_snapshots
            self._passive_detecting_requests[detector_label].rejected_marker_snapshots = \
                response.rejected_marker_snapshots
            self._passive_detecting_requests[detector_label].marker_snapshot_timestamp = \
                datetime.datetime.utcnow()  # TODO: This should come from the detector

    def handle_response_get_poses(
        self,
        response: GetPosesResponse
    ) -> None:
        if not self._is_solving:
            return
        self._tracked_target_poses.clear()
        self._renderer.clear_scene_objects()
        self._renderer.add_scene_object(  # Reference
            model_key=POSE_REPRESENTATIVE_MODEL,
            transform_to_world=Matrix4x4())
        table_rows: list[TrackingTableRow] = list()
        for pose in response.target_poses:
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
            self._renderer.add_scene_object(
                model_key=POSE_REPRESENTATIVE_MODEL,
                transform_to_world=pose.object_to_reference_matrix)
        for pose in response.detector_poses:
            table_row: TrackingTableRow = TrackingTableRow(
                target_id=pose.target_id,
                label=pose.target_id,
                x=pose.object_to_reference_matrix[0, 3],
                y=pose.object_to_reference_matrix[1, 3],
                z=pose.object_to_reference_matrix[2, 3])
            table_rows.append(table_row)
            self._tracked_target_poses.append(pose)
            self._renderer.add_scene_object(
                model_key=POSE_REPRESENTATIVE_MODEL,
                transform_to_world=pose.object_to_reference_matrix)
        self._tracking_table.update_contents(row_contents=table_rows)
        if len(table_rows) > 0:
            self._tracking_table.Enable(True)
        else:
            self._tracking_table.Enable(False)

    def handle_response_list_calibration_detector_resolutions(
        self,
        response: ListCalibrationDetectorResolutionsResponse
    ) -> None:
        self._calibrated_resolutions = response.detector_resolutions

    def handle_response_list_calibration_result_metadata(
        self,
        response: ListCalibrationResultMetadataResponse,
        detector_label: str
    ) -> None:
        if len(response.metadata_list) <= 0:
            self.status_message_source.enqueue_status_message(
                severity="error",
                message=f"No calibration was available for detector {detector_label}. No intrinsics will be set.")
            return
        newest_result_id: str = response.metadata_list[0].identifier  # placeholder, maybe
        newest_timestamp: datetime.datetime = datetime.datetime.min
        for result_metadata in response.metadata_list:
            timestamp: datetime.datetime = datetime.datetime.fromisoformat(result_metadata.timestamp_utc)
            if timestamp > newest_timestamp:
                newest_result_id = result_metadata.identifier
        self._detector_calibration_labels[detector_label] = newest_result_id

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
            elif isinstance(response, GetCalibrationResultResponse):
                self.handle_response_get_calibration_result(response=response)
                success = True
            elif isinstance(response, GetCapturePropertiesResponse):
                self.handle_response_get_capture_properties(
                    response=response,
                    detector_label=response_series.responder)
                success = True
            elif isinstance(response, GetMarkerSnapshotsResponse):
                self.handle_response_get_marker_snapshots(
                    response=response,
                    detector_label=response_series.responder)
            elif isinstance(response, GetPosesResponse):
                self.handle_response_get_poses(response=response)
                success = True
            elif isinstance(response, ListCalibrationDetectorResolutionsResponse):
                self.handle_response_list_calibration_detector_resolutions(response=response)
                success = True
            elif isinstance(response, ListCalibrationResultMetadataResponse):
                self.handle_response_list_calibration_result_metadata(
                    response=response,
                    detector_label=response_series.responder)
                success = True
            elif isinstance(response, ErrorResponse):
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
            calibrator_labels: list[str] = self._connector.get_connected_detector_labels()
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
        elif self._current_phase == ACTIVE_PHASE_STARTING_GET_RESOLUTIONS:
            self.status_message_source.enqueue_status_message(
                severity="debug",
                message="ACTIVE_PHASE_STARTING_GET_RESOLUTIONS complete")
            requests: list[MCastRequest] = list()
            for detector_label, image_resolution in self._detector_resolutions.items():
                detector_resolution: DetectorResolution = DetectorResolution(
                    detector_serial_identifier=detector_label,
                    image_resolution=image_resolution)
                if detector_resolution in self._calibrated_resolutions:
                    requests.append(
                        ListCalibrationResultMetadataRequest(
                            detector_serial_identifier=detector_resolution.detector_serial_identifier,
                            image_resolution=image_resolution))
                else:
                    self.status_message_source.enqueue_status_message(
                        severity="error",
                        message=f"No calibration available for detector {detector_label} "
                                f"at resolution {str(image_resolution)}. No intrinsics will be set.")
            calibrator_labels: list[str] = self._connector.get_connected_detector_labels()
            request_series: MCastRequestSeries = MCastRequestSeries(series=requests)
            self._active_request_ids.append(self._connector.request_series_push(
                connection_label=calibrator_labels[0],
                request_series=request_series))
            self._current_phase = ACTIVE_PHASE_STARTING_LIST_INTRINSICS
        elif self._current_phase == ACTIVE_PHASE_STARTING_LIST_INTRINSICS:
            self.status_message_source.enqueue_status_message(
                severity="debug",
                message="ACTIVE_PHASE_STARTING_LIST_INTRINSICS complete")
            requests: list[MCastRequest] = list()
            for detector_label, result_identifier in self._detector_calibration_labels.items():
                requests.append(GetCalibrationResultRequest(result_identifier=result_identifier))
            calibrator_labels: list[str] = self._connector.get_connected_detector_labels()
            request_series: MCastRequestSeries = MCastRequestSeries(series=requests)
            self._active_request_ids.append(self._connector.request_series_push(
                connection_label=calibrator_labels[0],
                request_series=request_series))
            self._current_phase = ACTIVE_PHASE_STARTING_GET_INTRINSICS
        elif self._current_phase == ACTIVE_PHASE_STARTING_GET_INTRINSICS:
            self.status_message_source.enqueue_status_message(
                severity="debug",
                message="ACTIVE_PHASE_STARTING_GET_INTRINSICS complete")
            requests: list[MCastRequest] = list()
            for detector_label, intrinsic_parameters in self._detector_intrinsics.items():
                requests.append(SetIntrinsicParametersRequest(
                    detector_label=detector_label,
                    intrinsic_parameters=intrinsic_parameters))
            requests.append(StartPoseSolverRequest())
            request_series: MCastRequestSeries = MCastRequestSeries(series=requests)
            self._active_request_ids.append(self._connector.request_series_push(
                connection_label=self._pose_solver_selector.selector.GetStringSelection(),
                request_series=request_series))
            self._current_phase = ACTIVE_PHASE_STARTING_FINAL
        elif self._current_phase == ACTIVE_PHASE_STARTING_FINAL:
            self.status_message_source.enqueue_status_message(
                severity="debug",
                message="ACTIVE_PHASE_STARTING_FINAL complete")
            for detector_label in self._detector_intrinsics.keys():
                self._passive_detecting_requests[detector_label] = PoseSolverPanel.PassiveDetectorRequest()
            self._is_solving = True
            self._current_phase = ACTIVE_PHASE_IDLE
        elif self._current_phase == ACTIVE_PHASE_STOPPING:
            self._tracking_table.update_contents(list())
            self._tracking_table.Enable(False)
            self._tracking_display_textbox.SetValue(str())
            self._tracking_display_textbox.Enable(False)
            self._current_phase = ACTIVE_PHASE_IDLE

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
        calibrator_labels: list[str] = self._connector.get_connected_detector_labels()
        if len(calibrator_labels) > 1:
            self.status_message_source.enqueue_status_message(
                severity="warning",
                message="Multiple calibrators are connected. "
                        "The first is being arbitrarily chosen for getting intrinsic parameters.")
        elif len(calibrator_labels) <= 0:
            self.status_message_source.enqueue_status_message(
                severity="error",
                message="No calibrators were found. Aborting tracking.")
            return
        self._detector_resolutions.clear()
        self._calibrated_resolutions.clear()
        self._detector_calibration_labels.clear()
        self._detector_intrinsics.clear()
        request_series: MCastRequestSeries = MCastRequestSeries(
            series=[ListCalibrationDetectorResolutionsRequest()])
        self._active_request_ids.append(self._connector.request_series_push(
            connection_label=calibrator_labels[0],
            request_series=request_series))
        detector_labels: list[str] = self._connector.get_connected_detector_labels()
        for detector_label in detector_labels:
            request_series: MCastRequestSeries = MCastRequestSeries(
                series=[StartCaptureRequest()])
            self._active_request_ids.append(self._connector.request_series_push(
                connection_label=detector_label,
                request_series=request_series))
        self._current_phase = ACTIVE_PHASE_STARTING_CAPTURE
        self._update_controls()

    def on_tracking_stop_pressed(self, _event: wx.CommandEvent) -> None:
        detector_labels: list[str] = self._connector.get_connected_detector_labels()
        for detector_label in detector_labels:
            request_series: MCastRequestSeries = MCastRequestSeries(
                series=[StopCaptureRequest()])
            self._active_request_ids.append(self._connector.request_series_push(
                connection_label=detector_label,
                request_series=request_series))
        request_series: MCastRequestSeries = MCastRequestSeries(series=[StopPoseSolverRequest()])
        selected_pose_solver_label: str = self._pose_solver_selector.selector.GetStringSelection()
        self._active_request_ids.append(self._connector.request_series_push(
            connection_label=selected_pose_solver_label,
            request_series=request_series))

        # Finish up any running passive tasks before we allow controls again
        for detecting_request in self._passive_detecting_requests.values():
            if detecting_request.request_id is not None:
                self._active_request_ids.append(detecting_request.request_id)
        self._passive_detecting_requests.clear()
        if self._passive_solving_request_id is not None:
            self._active_request_ids.append(self._passive_solving_request_id)
        self._passive_solving_request_id = None

        self._is_solving = False
        self._update_controls()
        self._current_phase = ACTIVE_PHASE_STOPPING

    def update_loop(self) -> None:
        super().update_loop()

        self._renderer.render()

        self._is_updating = True

        ui_needs_update: bool = False

        if self._is_solving:
            for detector_label, request_state in self._passive_detecting_requests.items():
                if request_state.request_id is not None:
                    _, request_state.request_id = self.update_request(request_id=request_state.request_id)
                if request_state.request_id is None:
                    if len(request_state.detected_marker_snapshots) > 0 or \
                       len(request_state.rejected_marker_snapshots) > 0:
                        detector_timestamp: str = request_state.marker_snapshot_timestamp.isoformat()
                        marker_request: AddMarkerCornersRequest = AddMarkerCornersRequest(
                            detected_marker_snapshots=request_state.detected_marker_snapshots,
                            rejected_marker_snapshots=request_state.rejected_marker_snapshots,
                            detector_label=detector_label,
                            detector_timestamp_utc_iso8601=detector_timestamp)
                        request_series: MCastRequestSeries = MCastRequestSeries(series=[marker_request])
                        selected_pose_solver_label: str = self._pose_solver_selector.selector.GetStringSelection()
                        request_state.request_id = self._connector.request_series_push(
                            connection_label=selected_pose_solver_label,
                            request_series=request_series)
                        request_state.detected_marker_snapshots.clear()
                        request_state.rejected_marker_snapshots.clear()
                    else:
                        request_series: MCastRequestSeries = MCastRequestSeries(series=[GetMarkerSnapshotsRequest()])
                        request_state.request_id = self._connector.request_series_push(
                            connection_label=detector_label,
                            request_series=request_series)
            if self._passive_solving_request_id is not None:
                _, self._passive_solving_request_id = \
                    self.update_request(request_id=self._passive_solving_request_id)
            if self._passive_solving_request_id is None:
                request_series: MCastRequestSeries = MCastRequestSeries(series=[GetPosesRequest()])
                selected_pose_solver_label: str = self._pose_solver_selector.selector.GetStringSelection()
                self._passive_solving_request_id = self._connector.request_series_push(
                    connection_label=selected_pose_solver_label,
                    request_series=request_series)

        # TODO: I think this can be moved to BasePanel class
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
        if len(self._active_request_ids) > 0:
            return  # We're waiting for something
        self._pose_solver_selector.Enable(True)
        self._reference_marker_id_spinbox.Enable(True)
        self._reference_target_submit_button.Enable(True)
        self._tracked_marker_id_spinbox.Enable(True)
        self._tracked_target_submit_button.Enable(True)
        if not self._is_solving:
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
                        message=f"Selected tracked target index {tracked_target_index} is out of bounds. Setting to None.")
                    self._tracking_table.set_selected_row_index(None)
                else:
                    self._tracking_display_textbox.Enable(True)

