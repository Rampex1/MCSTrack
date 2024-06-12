from pydantic import BaseModel, Field


class _Material(BaseModel):
    shader_label: str = Field()
    properties: dict[str, list[float] | float] = Field()


class _ModelPart(BaseModel):
    geometry_label: str = Field()
    material_label: str = Field()


class _Model(BaseModel):
    parts: list[_ModelPart] = Field()


class _Shader(BaseModel):
    vertex_shader_label: str = Field()
    fragment_shader_label: str = Field()
    property_list: list[str] = Field()


class FileIO:
    Material = _Material
    ModelPart = _ModelPart
    Model = _Model
    Shader = _Shader
