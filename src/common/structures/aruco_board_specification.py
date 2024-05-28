from pydantic import BaseModel, validator
from .vec3 import Vec3
import json, hjson

class BoardMarker(BaseModel):
    marker_id: int
    points: list[Vec3]

    @validator
    def check_points_length(cls, values):
        points = values.get('points')
        if points is not None and len(points) != 4:
            raise ValueError("The list of points must have exactly four elements")
        return values


class Board(BaseModel):
    board_markers: list[BoardMarker]

def read_file(input_filepath: str) -> Board:
    with open(input_filepath, 'r') as file:
        data = hjson.load(file)
    return Board(**data)


def write_file(output_filepath: str, output_board: Board) -> None:
    with open(output_filepath, 'w') as file:
        json.dump(output_board.as_dict(), file, indent=4)
