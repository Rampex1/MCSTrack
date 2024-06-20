class CharucoBoard:
    """
    Representation of the Charuco Board
    Marker 0 is at the top left, origin is at the bottom left.
    Positive x goes from left to right, positive y goes upwards, positive z goes out of the board
    """

    _board_marker_ids = list[int]
    _board_marker_positions = list[list[int]]

    def __init__(self):
        self._board_marker_ids = []
        self._board_marker_positions = []

    def _generate_board(self):
        for i in range(40):
            self._board_marker_ids.append(i)

        for marker_id in self._board_marker_ids:
            x_coords = [30.0, 70.0, 110.0, 150.0, 10.0, 50.0, 90.0, 130.0]
            x = x_coords[marker_id % 8]
            y = 190.0 - (marker_id // 4) * 20.0
            z = 0.0
            self._board_marker_positions.append([x, y, z])

    def get_ids(self):
        self._generate_board()
        return self._board_marker_ids

    def get_positions(self):
        self._generate_board()
        return self._board_marker_positions

