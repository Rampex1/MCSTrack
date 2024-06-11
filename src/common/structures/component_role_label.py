from typing import Final, Literal

COMPONENT_ROLE_LABEL_BOARD_BUILDER: Final[str] = "board_builder"
COMPONENT_ROLE_LABEL_CALIBRATOR: Final[str] = "calibrator"
COMPONENT_ROLE_LABEL_CONNECTOR: Final[str] = "connector"
COMPONENT_ROLE_LABEL_DETECTOR: Final[str] = "detector"
COMPONENT_ROLE_LABEL_POSE_SOLVER: Final[str] = "pose_solver"
ComponentRoleLabel = Literal["board_builder", "calibrator", "connector", "detector", "pose_solver"]
