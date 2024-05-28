from src.gui.panels import \
    BasePanel, \
    BoardBuilderPanel, \
    CalibratorPanel, \
    ConnectorPanel, \
    DetectorPanel, \
    PoseSolverPanel
from src.common import StatusMessageSource
from src.connector import Connector
import asyncio
import logging
import wx
import wxasync
from typing import Final


CONNECTOR_LABEL: Final[str] = "Connector"
DETECTOR_LABEL: Final[str] = "Detector"
CALIBRATOR_LABEL: Final[str] = "Calibrator"
BOARD_BUILDER_LABEL: Final[str] = "Board Builder"
POSE_SOLVER_LABEL: Final[str] = "Pose Solver"


# noinspection PyMethodMayBeStatic
class ControllerFrame(wx.Frame):

    _status_message_source: StatusMessageSource
    _connector: Connector

    _notebook: wx.Notebook
    _connector_panel: ConnectorPanel
    _detector_panel: DetectorPanel
    _calibrator_panel: CalibratorPanel
    _pose_solver_panel: PoseSolverPanel

    def __init__(
        self,
        connector: Connector,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._status_message_source = StatusMessageSource(
            source_label="gui",
            send_to_logger=True)
        self._connector = connector

        self.SetMinSize(wx.Size(640, 480))

        frame_panel = wx.Panel(self)
        self._notebook = wx.Notebook(frame_panel)
        frame_panel_sizer: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)
        frame_panel_sizer.Add(
            self._notebook,
            wx.SizerFlags(1).Expand())
        frame_panel.SetSizerAndFit(frame_panel_sizer)

        self._connector_panel = ConnectorPanel(
            parent=self._notebook,
            connector=self._connector,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._connector_panel,
            text=CONNECTOR_LABEL,
            select=True)
        self._connector_panel.panel_is_selected = True

        self._detector_panel = DetectorPanel(
            parent=self._notebook,
            connector=self._connector,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._detector_panel,
            text=DETECTOR_LABEL,
            select=False)

        self._calibrator_panel = CalibratorPanel(
            parent=self._notebook,
            connector=self._connector,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._calibrator_panel,
            text=CALIBRATOR_LABEL,
            select=False)

        self._board_builder_panel = BoardBuilderPanel(
            parent=self._notebook,
            connector=self._connector,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._board_builder_panel,
            text=BOARD_BUILDER_LABEL,
            select=False)

        self._pose_solver_panel = PoseSolverPanel(
            parent=self._notebook,
            connector=self._connector,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._pose_solver_panel,
            text=POSE_SOLVER_LABEL,
            select=False)

        self._notebook.Bind(
            event=wx.EVT_BOOKCTRL_PAGE_CHANGED,
            handler=self.on_page_changed)

        self.CreateStatusBar()
        self.SetStatusText("")

    def on_page_changed(self, event: wx.BookCtrlEvent):
        pages: list[BasePanel] = [
            self._connector_panel,
            self._detector_panel,
            self._calibrator_panel,
            self._pose_solver_panel]
        for page in pages:
            page_index: int = self._notebook.FindPage(page)
            if page_index == event.GetOldSelection():
                page.on_page_deselect()
                break
        for page in pages:
            page_index: int = self._notebook.FindPage(page)
            if page_index == event.GetSelection():
                page.on_page_select()
                break


async def connector_frame_repeat(connector: Connector):
    # noinspection PyBroadException
    try:
        await connector.do_update_frames_for_connections()
    except Exception as e:
        connector.add_status_message(
            severity="error",
            message=f"Exception occurred in connector loop: {str(e)}")
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    event_loop.create_task(connector_frame_repeat(connector=connector))


async def main():
    logging.basicConfig(level=logging.INFO)

    connector = Connector(
        serial_identifier="connector",
        send_status_messages_to_logger=True)
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    event_loop.create_task(connector_frame_repeat(connector=connector))

    app: wxasync.WxAsyncApp = wxasync.WxAsyncApp()
    frame: ControllerFrame = ControllerFrame(
        connector=connector,
        parent=None,
        title="MCAST Controller")
    frame.Show()
    await app.MainLoop()


if __name__ == "__main__":
    asyncio.run(main())
