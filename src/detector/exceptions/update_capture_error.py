from src.common.exceptions import MCastError
from src.common.structures import SeverityLabel


class UpdateCaptureError(MCastError):
    severity: SeverityLabel
    message: str

    def __init__(self, *args, severity: SeverityLabel, message: str):
        super().__init__(*args)
        self.severity = severity
        self.message = message
