from .parameters import \
    ParameterCheckbox, \
    ParameterSelector, \
    ParameterSpinboxFloat, \
    ParameterSpinboxInteger, \
    ParameterText
from src.common import \
    ErrorResponse, \
    MCTResponse, \
    StatusMessageSource
from typing import Final
import wx


_UPDATE_INTERVAL_MILLISECONDS: Final[int] = 16


class BasePanel(wx.Panel):

    panel_is_selected: bool
    status_message_source: StatusMessageSource
    DEFAULT_SPACING_PX_VERTICAL: Final[int] = 4
    DEFAULT_SPACING_PX_LINE_TOP_BOTTOM: Final[int] = 8

    _update_loop_running: bool

    def __init__(
        self,
        parent: wx.Window,
        status_message_source: StatusMessageSource,
        name: str
    ):
        super().__init__(parent=parent, name=name)
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

    def handle_unknown_response(
        self,
        response: MCTResponse
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
        if not self.panel_is_selected:
            self._update_loop_running = False
            return
        wx.CallLater(_UPDATE_INTERVAL_MILLISECONDS, self.update_loop)

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
        selectable_values: list[str],
        value: str | None = None
    ) -> ParameterSelector:
        selector = ParameterSelector(
            parent=parent,
            label=label,
            selectable_values=selectable_values,
            value=value)
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
        step_value: float,
        digit_count: int = 2
    ) -> ParameterSpinboxFloat:
        spinbox = ParameterSpinboxFloat(
            parent=parent,
            label=label,
            minimum_value=minimum_value,
            maximum_value=maximum_value,
            initial_value=initial_value,
            step_value=step_value,
            digit_count=digit_count)
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
        initial_value: int,
        step_value: int = 1
    ) -> ParameterSpinboxInteger:
        spinbox = ParameterSpinboxInteger(
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

