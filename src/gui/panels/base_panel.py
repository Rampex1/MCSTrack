from .parameters import \
    ParameterCheckbox, \
    ParameterSelector, \
    ParameterSpinboxFloat, \
    ParameterSpinboxInteger, \
    ParameterText
from src.common import \
    ErrorResponse, \
    MCastResponse, \
    MCastResponseSeries, \
    StatusMessageSource
from src.connector import \
    Connector
from typing import Optional
from typing import Final
import uuid
import wx


_UPDATE_INTERVAL_MILLISECONDS: Final[int] = 16


class BasePanel(wx.Panel):

    panel_is_selected: bool
    status_message_source: StatusMessageSource
    DEFAULT_SPACING_PX_VERTICAL: Final[int] = 4
    DEFAULT_SPACING_PX_LINE_TOP_BOTTOM: Final[int] = 8

    _update_loop_running: bool

    _connector: Connector

    def __init__(
        self,
        parent: wx.Window,
        connector: Connector,
        status_message_source: StatusMessageSource,
        name: str
    ):
        super().__init__(parent=parent, name=name)
        self._connector = connector
        self.panel_is_selected = False
        self.status_message_source = status_message_source

        self._update_loop_running = True
        wx.CallLater(_UPDATE_INTERVAL_MILLISECONDS, self.update_loop)

    def handle_error_response(
        self,
        response: ErrorResponse
    ):
        self.status_message_source.enqueue_status_message(
            severity="error",
            message=f"Received error: {response.message}")

    def handle_response_series(
        self,
        response_series: MCastResponseSeries,
        task_description: Optional[str] = None,
        expected_response_count: Optional[int] = None
    ) -> bool:  # return False if errors occurred
        if expected_response_count is not None:
            response_count: int = len(response_series.series)
            task_text: str = str()
            if task_description is not None:
                task_text = f" during {task_description}"
            if response_count < expected_response_count:
                self.status_message_source.enqueue_status_message(
                    severity="warning",
                    message=f"Received a response series{task_text}, "
                            f"but it contained fewer responses ({response_count}) "
                            f"than expected ({expected_response_count}).")
            elif response_count > expected_response_count:
                self.status_message_source.enqueue_status_message(
                    severity="warning",
                    message=f"Received a response series{task_text}, "
                            f"but it contained more responses ({response_count}) "
                            f"than expected ({expected_response_count}).")
        return True

    def handle_unknown_response(
        self,
        response: MCastResponse
    ):
        self.status_message_source.enqueue_status_message(
            severity="error",
            message=f"Received unexpected response: {str(type(response))}")

    def on_page_select(self):
        self.status_message_source.enqueue_status_message(
            severity="debug",
            message=f"{self.GetName()} on_page_select")
        self.panel_is_selected = True
        if not self._update_loop_running:
            self._update_loop_running = True
            self.update_loop()

    def on_page_deselect(self):
        self.status_message_source.enqueue_status_message(
            severity="debug",
            message=f"{self.GetName()} on_page_deselect")
        self.panel_is_selected = False

    def update_loop(self) -> None:
        """
        Overload for anything that should be updated approximately once per GUI frame
        """
        self._connector.update_loop()
        if not self.panel_is_selected:
            self._update_loop_running = False
            return
        wx.CallLater(_UPDATE_INTERVAL_MILLISECONDS, self.update_loop)

    def update_request(
        self,
        request_id: uuid.UUID,
        task_description: Optional[str] = None,
        expected_response_count: Optional[int] = None
    ) -> (bool, uuid.UUID | None):
        """
        Returns a tuple of:
        - success at handling the response (False if no response has been received)
        - value that request_id shall take for subsequent iterations (None means a response series has been received)
        """

        response_series: MCastResponseSeries | None = self._connector.response_series_pop(
            request_series_id=request_id)
        if response_series is None:
            return False, request_id  # try again next loop

        success: bool = self.handle_response_series(
            response_series=response_series,
            task_description=task_description,
            expected_response_count=expected_response_count)
        return success, None  # We've handled the request, request_id can be set to None

    # -------------------------------------------------------------------------------------

    @staticmethod
    def add_control_button(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str
    ) -> wx.Button:
        button = wx.Button(
            parent=parent,
            label=label)
        sizer.Add(
            window=button,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return button

    @staticmethod
    def add_control_checkbox(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str,
        value: bool = False
    ) -> ParameterCheckbox:
        checkbox = ParameterCheckbox(
            parent=parent,
            label=label,
            value=value)
        sizer.Add(
            window=checkbox,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return checkbox

    @staticmethod
    def add_control_selector(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str,
        selectable_values: list[str]
    ) -> ParameterSelector:
        selector = ParameterSelector(
            parent=parent,
            label=label,
            selectable_values=selectable_values)
        sizer.Add(
            window=selector,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return selector

    @staticmethod
    def add_control_spinbox_float(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str,
        minimum_value: float,
        maximum_value: float,
        initial_value: float,
        step_value: float
    ) -> ParameterSpinboxFloat:
        spinbox = ParameterSpinboxFloat(
            parent=parent,
            label=label,
            minimum_value=minimum_value,
            maximum_value=maximum_value,
            initial_value=initial_value,
            step_value=step_value)
        sizer.Add(
            window=spinbox,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return spinbox

    @staticmethod
    def add_control_spinbox_integer(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str,
        minimum_value: int,
        maximum_value: int,
        initial_value: int
    ) -> ParameterSpinboxInteger:
        spinbox = ParameterSpinboxInteger(
            parent=parent,
            label=label,
            minimum_value=minimum_value,
            maximum_value=maximum_value,
            initial_value=initial_value)
        sizer.Add(
            window=spinbox,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return spinbox

    @staticmethod
    def add_control_text_input(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str,
        value: str = str()
    ) -> ParameterText:
        textbox = ParameterText(
            parent=parent,
            label=label,
            value=value)
        sizer.Add(
            window=textbox,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return textbox

    @staticmethod
    def add_horizontal_line_to_spacer(
        parent: wx.Window,
        sizer: wx.BoxSizer
    ) -> None:
        line: wx.StaticLine = wx.StaticLine(parent=parent)
        sizer.Add(
            window=line,
            flags=wx.SizerFlags(0).Border(
                direction=(wx.TOP | wx.BOTTOM),
                borderinpixels=BasePanel.DEFAULT_SPACING_PX_LINE_TOP_BOTTOM).Expand())

    @staticmethod
    def add_text_label(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str,
        font_size_delta: int | None = None,
        bold: bool | None = None
    ) -> None:
        line: wx.StaticText = wx.StaticText(parent=parent, label=label)
        font: wx.Font = line.GetFont()
        if font_size_delta is not None:
            font.SetPointSize(pointSize=font.GetPointSize() + font_size_delta)
        if bold is True:
            font.MakeBold()
        line.SetFont(font)
        sizer.Add(
            window=line,
            flags=wx.SizerFlags(0).Border(
                direction=(wx.TOP | wx.BOTTOM),
                borderinpixels=BasePanel.DEFAULT_SPACING_PX_LINE_TOP_BOTTOM).Expand())

    @staticmethod
    def add_toggle_button(
        parent: wx.Window,
        sizer: wx.BoxSizer,
        label: str
    ) -> wx.ToggleButton:
        toggle_button = wx.ToggleButton(
            parent=parent,
            label=label)
        sizer.Add(
            window=toggle_button,
            flags=wx.SizerFlags(0).Expand())
        sizer.AddSpacer(size=BasePanel.DEFAULT_SPACING_PX_VERTICAL)
        return toggle_button

