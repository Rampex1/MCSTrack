from src.common import MCastRequest
from pydantic import Field
from typing import List


class LocateReferenceMarkersRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "locate_reference_markers_parameters"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    ids: List[int] = Field()
    corners: List[List[List[float]]] = Field()
