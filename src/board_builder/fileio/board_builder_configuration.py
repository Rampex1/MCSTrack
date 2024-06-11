from pydantic import BaseModel, Field


class BoardBuilderConfiguration(BaseModel):
    serial_identifier: str = Field()
