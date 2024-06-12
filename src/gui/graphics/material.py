from .constants import Constants
from .shader import Shader
import logging
from OpenGL.GL import *
import numpy


logger = logging.getLogger(__name__)


class Material:
    """
    Like other graphics classes, Material is closely tied to an OpenGL context.
    The user shall ensure that the provided Shader is loaded into the context in which this Material is used.
    It is the user's responsibility to make sure that data do not get mixed between contexts.
    """

    _shader: Shader
    _properties: dict[str, numpy.array]

    def __init__(
        self,
        shader: Shader,
        properties: dict[str, list]
    ):
        self._shader = shader
        self._properties = dict()
        for property_key, property_value in properties.items():
            if isinstance(property_value, list):
                self._properties[property_key] = numpy.array(property_value, dtype="float32")
            elif isinstance(property_value, float):
                self._properties[property_key] = numpy.array([property_value], dtype="float32")
            else:
                raise RuntimeError(
                    "Unsupported property type received in material, " +
                    "got " + str(property_value.__class__))

    def get_shader(self) -> Shader:
        return self._shader

    def use(self):
        shader_program = self._shader.get_shader_program()
        glUseProgram(shader_program)
        property_key: str
        property_value: numpy.array
        for property_key, property_value in self._properties.items():
            uniform_location: GLint = glGetUniformLocation(shader_program, property_key)
            if uniform_location == -1:
                raise RuntimeError(
                    "Unexpected property for Material: " + property_key + ".\n"
                    "Check the material file and ensure the material properties match those in the shader?")
            if property_value.shape == (1,) and property_value.dtype == numpy.dtype("float32"):
                glUniform1fv(uniform_location, 1, property_value)
            elif property_value.shape == (3,) and property_value.dtype == numpy.dtype("float32"):
                glUniform3fv(uniform_location, 1, property_value)
            elif property_value.shape == (4,) and property_value.dtype == numpy.dtype("float32"):
                glUniform4fv(uniform_location, 1, property_value)
            elif property_value.shape == (4, 4) and property_value.dtype == numpy.dtype("float32"):
                glUniformMatrix4fv(uniform_location, 1, Constants.GL_TRANSPOSE_MATRIX, property_value)
            else:
                raise RuntimeError(
                    "Unsupported type of numpy array: " +
                    str(property_value.shape) + ", " + str(property_value.dtype))
