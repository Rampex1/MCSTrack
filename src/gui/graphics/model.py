from .constants import Constants
from .material import Material
from .shader import Shader
import logging
import numpy
from OpenGL.GL import *
from scipy.spatial.transform import Rotation
from typing import Final


logger = logging.getLogger(__name__)
_VERTEX_COUNT_PER_FACE: Final[int] = 3


class Model:
    """
    Like other graphics classes, Model and ModelPart are closely tied to an OpenGL context.
    After load_into_opengl(), the Model and ModelPart are permanently bound and can be used only with that context.
    Please note that the Material associated with the various ModelPart will also be loaded if not already done so.
    It is the user's responsibility to make sure that data do not get mixed between contexts.
    """

    class ModelPart:
        # raw data
        _vertices: numpy.ndarray | None
        _normals: numpy.ndarray | None
        _triangles: list[list[int]] | None

        _loaded_into_opengl: bool

        # Loaded OpenGL data
        _vao: GLuint
        _vbo: GLuint
        _vertex_count_in_part: int

        # OpenGL data provided by the user
        _material: Material

        def __init__(
            self,
            vertices: numpy.array,
            normals: numpy.array,
            triangles: list[list[int]],  # each sublist is length 3, indices of vertices
            material: Material
        ):
            self._vertices = vertices
            self._normals = normals
            self._triangles = triangles
            self._loaded_into_opengl = False
            self._material = material

        def get_shader(self) -> Shader:
            return self._material.get_shader()

        def is_loaded_into_opengl(self) -> bool:
            return self._loaded_into_opengl

        def load_into_opengl(self) -> None:
            if self._loaded_into_opengl:
                raise RuntimeError("load_into_opengl() called repeatedly on the same ModelPart.")

            # Definition of this object in GPU will be associated with this VAO
            self._vao = GLuint()
            glGenVertexArrays(1, self._vao)
            glBindVertexArray(self._vao)

            # Store the vertices and normals in a VBO
            face_count = len(self._triangles)
            self._vertex_count_in_part = face_count * _VERTEX_COUNT_PER_FACE
            floats_per_vertex = 6
            buffer = numpy.empty(floats_per_vertex * self._vertex_count_in_part, dtype="float32")
            for face_index, face in enumerate(self._triangles):
                for buffer_vertex_index, face_vertex_index in enumerate(face):
                    base_index_for_vertex = \
                        ((face_index * _VERTEX_COUNT_PER_FACE) + buffer_vertex_index) * floats_per_vertex
                    position_begin_index = base_index_for_vertex
                    position_end_index = base_index_for_vertex + 3
                    normal_begin_index = position_end_index
                    normal_end_index = normal_begin_index + 3
                    buffer[position_begin_index:position_end_index] = self._vertices[face_vertex_index]
                    buffer[normal_begin_index:normal_end_index] = self._normals[face_index]

            self._vbo = GLuint(0)
            glGenBuffers(1, self._vbo)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, buffer, GL_STATIC_DRAW)

            size_of_float = 4  # float32, 32 bits is 4 bytes
            size_of_float_3 = size_of_float * 3
            size_of_float_6 = size_of_float * 6

            attribute_index_position = 0
            glVertexAttribPointer(
                attribute_index_position,
                3, GL_FLOAT, GL_FALSE,
                size_of_float_6,
                ctypes.c_void_p(0))
            glEnableVertexAttribArray(attribute_index_position)

            attribute_index_normal = 1
            glVertexAttribPointer(
                attribute_index_normal,
                3, GL_FLOAT, GL_FALSE,
                size_of_float_6,
                ctypes.c_void_p(size_of_float_3))
            glEnableVertexAttribArray(attribute_index_normal)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

            self._loaded_into_opengl = True

            if not self.get_shader().is_loaded_into_opengl():
                self.get_shader().load_into_opengl()

            # These data aren't needed anymore - release them to free up memory
            self._vertices = None
            self._normals = None
            self._triangles = None

        def draw(
            self,
            translation: list[float],
            rotation_quaternion: list[float]
        ) -> None:
            if not self._loaded_into_opengl:
                logger.warning("draw() called but ModelPart has not been loaded into OpenGL. Will do so now.")
                self.load_into_opengl()

            assert (isinstance(translation, list) and len(translation) == 3)
            assert (isinstance(rotation_quaternion, list) and len(rotation_quaternion) == 4)

            self._material.use()

            object_to_world_matrix = numpy.identity(4, dtype="float32")
            object_to_world_matrix[0:3, 0:3] = Rotation.from_quat(rotation_quaternion).as_matrix()
            object_to_world_matrix[0:3, 3] = translation
            object_to_world_location = glGetUniformLocation(
                self._material.get_shader().get_shader_program(), Constants.GL_OBJECT_TO_WORLD_PROPERTY_KEY)
            glUniformMatrix4fv(object_to_world_location, 1, Constants.GL_TRANSPOSE_MATRIX, object_to_world_matrix)

            glBindVertexArray(self._vao)
            glDrawArrays(GL_TRIANGLES, 0, self._vertex_count_in_part)
            glBindVertexArray(0)

    _parts: list[ModelPart]
    _loaded_into_opengl: bool

    def __init__(
        self,
        parts: list[ModelPart]
    ):
        self._parts = parts
        self._loaded_into_opengl = False

    def load_into_opengl(self):
        if self._loaded_into_opengl:
            raise RuntimeError("load_into_opengl() called repeatedly on the same Model.")
        for part in self._parts:
            part.load_into_opengl()
        self._loaded_into_opengl = True

    def draw(
        self,
        translation: list[float],
        rotation_quaternion: list[float]
    ):
        if not self._loaded_into_opengl:
            logger.warning("draw() called but Model has not been loaded into OpenGL. Will do so now.")
            self.load_into_opengl()
        assert (isinstance(translation, list) and len(translation) == 3)
        assert (isinstance(rotation_quaternion, list) and len(rotation_quaternion) == 4)
        for part in self._parts:
            part.draw(translation, rotation_quaternion)

    def get_shaders(self) -> set[Shader]:
        return {part.get_shader() for part in self._parts}
