from src.common.structures.component_role_label import ComponentRoleLabel
from ipaddress import IPv4Address
from pydantic import BaseModel, Field


class ComponentConnectionStatic(BaseModel):
    """
    Information used to establish a connection and display information about said connection,
    but nothing that would change without a user's explicit input (e.g. status etc).
    """
    label: str = Field()
    role: ComponentRoleLabel = Field()
    ip_address: IPv4Address = Field()
    port: int = Field()
