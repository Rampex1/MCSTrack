import wx


class ImagePanel(wx.Panel):
    _bitmap: wx.Bitmap

    def __init__(
        self,
        parent: wx.Window
    ):
        super().__init__(parent=parent)
        self._bitmap = wx.Bitmap(width=1, height=1)
        self.Bind(
            event=wx.EVT_PAINT,
            handler=self.on_paint)

    def on_paint(
        self,
        _event: wx.PaintEvent
    ):
        device_context = wx.PaintDC(self)
        self.render(device_context=device_context)
        self.render(device_context=device_context)

    def paint(
        self
    ):
        device_context = wx.ClientDC(self)
        self.render(device_context=device_context)

    def render(
        self,
        device_context: wx.DC
    ):
        device_context.DrawBitmap(
            bitmap=self._bitmap,
            x=0,
            y=0,
            useMask=False)

    def set_bitmap(
        self,
        bitmap: wx.Bitmap
    ):
        self._bitmap = bitmap
