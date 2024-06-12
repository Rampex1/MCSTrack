from typing import Final, Literal

COMPONENT_ROLE_LABEL_CONNECTOR: Final[str] = "connector"
COMPONENT_ROLE_LABEL_DETECTOR: Final[str] = "detector"
COMPONENT_ROLE_LABEL_POSE_SOLVER: Final[str] = "pose_solver"
ComponentRoleLabel = Literal["connector", "detector", "pose_solver"]
