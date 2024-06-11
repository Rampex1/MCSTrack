from src.common import MCastRequest
from pydantic import Field


class StartBoardBuilderRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "start_board_builder"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
