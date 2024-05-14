from pydantic import BaseModel

from typing import List
from pydantic import BaseModel
from .vec3 import Vec3

class BoardMarker(BaseModel):
    marker_id: int
    points: List[Vec3]


class ArucoBoardSpecification(BaseModel):
    board_markers: List[BoardMarker]


def read_file(input_filepath: str) -> ArucoBoardSpecification:
    return ArucoBoardSpecification.parse_file(input_filepath)


def write_file(output_filepath: str, output_board: ArucoBoardSpecification) -> None:
    output_board.write_file(output_filepath)
