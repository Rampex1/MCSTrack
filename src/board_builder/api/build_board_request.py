from src.common import MCastRequest
from pydantic import Field
from typing import List


class BuildBoardRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "build_board"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    ids: List[int] = Field()
    corners: List[List[List[float]]] = Field()
