from src.gui.panels import \
    BasePanel, \
    BoardBuilderPanel, \
    CalibratorPanel, \
    ControllerPanel, \
    DetectorPanel, \
    PoseSolverPanel
from src.common import StatusMessageSource
from src.controller import MCTController
import asyncio
import logging
import wx
import wxasync
from typing import Final


CONTROLLER_LABEL: Final[str] = "Controller"
DETECTOR_LABEL: Final[str] = "Detector"
CALIBRATOR_LABEL: Final[str] = "Calibrator"
BOARD_BUILDER_LABEL: Final[str] = "Board Builder"
POSE_SOLVER_LABEL: Final[str] = "Pose Solver"


# noinspection PyMethodMayBeStatic
class ControllerFrame(wx.Frame):

    _status_message_source: StatusMessageSource
    _controller: MCTController

    _notebook: wx.Notebook
    _controller_panel: ControllerPanel
    _detector_panel: DetectorPanel
    _calibrator_panel: CalibratorPanel
    _board_builder_panel: BoardBuilderPanel
    _pose_solver_panel: PoseSolverPanel

    def __init__(
        self,
        controller: MCTController,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._status_message_source = StatusMessageSource(
            source_label="gui",
            send_to_logger=True)
        self._controller = controller

        self.SetMinSize(wx.Size(800, 600))

        frame_panel = wx.Panel(self)
        self._notebook = wx.Notebook(frame_panel)
        frame_panel_sizer: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)
        frame_panel_sizer.Add(
            self._notebook,
            wx.SizerFlags(1).Expand())
        frame_panel.SetSizerAndFit(frame_panel_sizer)

        self._controller_panel = ControllerPanel(
            parent=self._notebook,
            controller=self._controller,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._controller_panel,
            text=CONTROLLER_LABEL,
            select=True)
        self._controller_panel.panel_is_selected = True

        self._detector_panel = DetectorPanel(
            parent=self._notebook,
            controller=self._controller,
            status_message_source=self._status_message_source)
        self._notebook.AddPage(
            page=self._detector_panel,
            text=DETECTOR_LABEL,
            select=False)

        self._calibrator_panel = CalibratorPanel(
            parent=self._notebook,
            controller=self._controller,
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
            controller=self._controller,
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
            self._controller_panel,
            self._detector_panel,
            self._calibrator_panel,
            self._board_builder_panel,
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


async def controller_frame_repeat(controller: MCTController):
    # noinspection PyBroadException
    try:
        await controller.update()
    except Exception as e:
        controller.add_status_message(
            severity="error",
            message=f"Exception occurred in controller loop: {str(e)}")
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    event_loop.create_task(controller_frame_repeat(controller=controller))


async def main():
    logging.basicConfig(level=logging.INFO)

    controller = MCTController(
        serial_identifier="controller",
        send_status_messages_to_logger=True)
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    event_loop.create_task(controller_frame_repeat(controller=controller))

    app: wxasync.WxAsyncApp = wxasync.WxAsyncApp()
    frame: ControllerFrame = ControllerFrame(
        controller=controller,
        parent=None,
        title="MCT Controller")
    frame.Show()
    await app.MainLoop()


if __name__ == "__main__":
    asyncio.run(main())
