import logging
from OpenGL.GL import *
from typing import List


logger = logging.getLogger(__name__)


class Shader:
    """
    Like other graphics classes, Shader is closely tied to an OpenGL context.
    After load_into_opengl(), the Shader is permanently bound and can be used only with that context.
    It is the user's responsibility to make sure that data do not get mixed between contexts.
    """

    # raw data
    _vertex_shader_code: str | None
    _fragment_shader_code: str | None
    _property_list: List[str]  # Not currently used, but could be used in the future for error checking in Material...?

    _loaded_into_opengl: bool

    # Loaded OpenGL data
    _shader_program: GLuint

    def __init__(
        self,
        vertex_shader_code: str,
        fragment_shader_code: str,
        property_list: List[str]
    ):
        self._vertex_shader_code = vertex_shader_code
        self._fragment_shader_code = fragment_shader_code
        self._property_list = property_list
        self._loaded_into_opengl = False

    def is_loaded_into_opengl(self) -> bool:
        return self._loaded_into_opengl

    def load_into_opengl(self) -> None:
        if self._loaded_into_opengl:
            logger.error("load_into_opengl() called repeatedly on the same Shader.")
            return

        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, self._vertex_shader_code)
        glCompileShader(vertex_shader)
        vertex_shader_compile_status = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
        if vertex_shader_compile_status == 0:
            message = glGetShaderInfoLog(vertex_shader).decode('utf-8')
            raise RuntimeError("Vertex shader compiler error: " + message)

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, self._fragment_shader_code)
        glCompileShader(fragment_shader)
        fragment_shader_compile_status = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
        if fragment_shader_compile_status == 0:
            message = glGetShaderInfoLog(fragment_shader).decode('utf-8')
            raise RuntimeError("Fragment shader compiler error: " + message)

        self._shader_program = glCreateProgram()
        glAttachShader(self._shader_program, vertex_shader)
        glAttachShader(self._shader_program, fragment_shader)
        glLinkProgram(self._shader_program)
        shader_program_link_status = glGetProgramiv(self._shader_program, GL_LINK_STATUS)
        if shader_program_link_status == 0:
            message = glGetProgramInfoLog(self._shader_program).decode('utf-8')
            raise RuntimeError("Shader program link error: " + message)

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        self._loaded_into_opengl = True

        # These data aren't needed anymore - release them to free up memory
        self._vertex_shader_code = None
        self._fragment_shader_code = None

    # Note TV 2024-05-24:
    #     This was in legacy code but commented. It might be safe to re-enable it, z
    # def __del__(self):
    #     glDeleteProgram(self._shader_program)

    def get_shader_program(self) -> GLuint:
        if not self.is_loaded_into_opengl():
            logger.warning(
                "get_shader_program() called but Material's Shader has not been loaded into OpenGL. Will do so now.")
            self.load_into_opengl()
        return self._shader_program
