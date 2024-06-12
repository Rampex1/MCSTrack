from src.common.structures import Matrix4x4
from src.gui.graphics import Constants, FileIO, Material, Model, Shader
import datetime
import hjson
import numpy
from OpenGL.GL import *
import os
from scipy.spatial.transform import Rotation
import stl
from typing import Final
import wx
from wx.glcanvas import GLCanvas, GLContext


_UP_DIRECTION: Final[list[float]] = [0.0, 0.0, 1.0]
_FORWARD_DIRECTION: Final[list[float]] = [0.0, 1.0, 0.0]
_VIEWPORT_FOV_DEGREES_MINIMUM: Final[float] = 10.0
_VIEWPORT_FOV_DEGREES_MAXIMUM: Final[float] = 120.0
_PERSPECTIVE_ALTITUDE_DEGREES_MAXIMUM: Final[float] = 89.5  # Clamped
_PERSPECTIVE_ALTITUDE_DEGREES_MINIMUM: Final[float] = -89.5  # Clamped
_PERSPECTIVE_AZIMUTH_DEGREES_MAXIMUM: Final[float] = 180.0  # Wraps around
_PERSPECTIVE_AZIMUTH_DEGREES_MINIMUM: Final[float] = -180.0  # Wraps around
_INPUT_MAXIMUM_MOUSE_SPEED_PIXELS_PER_SECOND: Final[float] = 10000.0
_INPUT_MAXIMUM_SCROLL_SPEED_UNITS_PER_SECOND: Final[float] = 120.0
_INPUT_MAXIMUM_ROTATION_SPEED_DEGREES_PER_SECOND: Final[float] = 9000.0
_INPUT_MAXIMUM_ZOOM_SPEED_DEGREES_PER_SECOND: Final[float] = 10.0
_INPUT_MAXIMUM_LATERAL_SPEED_MILLIMETERS_PER_SECOND: Final[float] = 10.0
_INPUT_MAXIMUM_DOLLY_SPEED_MILLIMETERS_PER_SECOND: Final[float] = 25.0  # forward/backward
_EPSILON: Final[float] = 0.0001


class GraphicsRenderer(GLCanvas, wx.Window):

    class SceneObject:
        model: Model
        transform_to_world: Matrix4x4

        def __init__(
            self,
            model: Model,
            transform_to_world: Matrix4x4
        ):
            self.model = model
            self.transform_to_world = transform_to_world

    _context: GLContext
    _last_render_datetime_utc: datetime.datetime  # drawing to screen
    _last_update_datetime_utc: datetime.datetime  # handling user events, regular tasks, etc
    _shaders: set[Shader]
    _model_dictionary: dict[str, Model]
    _scene_objects: list[SceneObject]

    # viewport
    _fov_degrees: float
    _near_plane_distance: float
    _far_plane_distance: float
    _background_color: list[float]  # length 4

    # perspective
    _perspective_target: list[float]
    _perspective_distance: float
    _perspective_azimuth_degrees: float
    _perspective_altitude_degrees: float
    _current_view_to_world_matrix: numpy.array

    # input
    _mouse_button_left_is_down: bool
    _mouse_button_middle_is_down: bool
    _mouse_button_right_is_down: bool
    _current_mouse_x_pixels: float
    _current_mouse_y_pixels: float
    _last_mouse_x_pixels: float
    _last_mouse_y_pixels: float

    def __init__(
        self,
        parent: wx.Window
    ):
        super().__init__(parent=parent)

        self._fov_degrees = 45.0
        self._near_plane_distance = 1.0
        self._far_plane_distance = 10000.0
        self._background_color = [0.4, 0.45, 0.5, 1.0]

        self._perspective_target = [0.0, 0.0, 0.0]
        self._perspective_distance = 400.0
        self._perspective_azimuth_degrees = 0.0
        self._perspective_altitude_degrees = 45.0
        self._current_view_to_world_matrix = numpy.identity(4, dtype="float32")

        self._mouse_button_left_is_down = False
        self._mouse_button_middle_is_down = False
        self._mouse_button_right_is_down = False
        self._current_mouse_x_pixels = 0.0
        self._current_mouse_y_pixels = 0.0
        self._last_mouse_x_pixels = 0.0
        self._last_mouse_y_pixels = 0.0

        # We'll let the caller manage when to render rather than the wx event system.
        # allows things like regular drawing intervals (update loop),
        # perhaps avoid uncertainties about when the paint event will trigger...
        self.Bind(
            event=wx.EVT_LEFT_DOWN,
            handler=self._on_mouse_button_left_down)
        self.Bind(
            event=wx.EVT_LEFT_UP,
            handler=self._on_mouse_button_left_up)
        self.Bind(
            event=wx.EVT_MIDDLE_DOWN,
            handler=self._on_mouse_button_middle_down)
        self.Bind(
            event=wx.EVT_MIDDLE_UP,
            handler=self._on_mouse_button_middle_up)
        self.Bind(
            event=wx.EVT_RIGHT_DOWN,
            handler=self._on_mouse_button_right_down)
        self.Bind(
            event=wx.EVT_RIGHT_UP,
            handler=self._on_mouse_button_right_up)
        self.Bind(
            event=wx.EVT_MOTION,
            handler=self._on_mouse_moved)
        self.Bind(
            event=wx.EVT_MOUSEWHEEL,
            handler=self._on_mouse_wheel)
        self.Bind(
            event=wx.EVT_SIZE,
            handler=self._on_resize)

        self._context = GLContext(win=self)
        self._last_render_datetime_utc = datetime.datetime.utcnow()
        self._last_update_datetime_utc = datetime.datetime.utcnow()
        self._shaders = set()
        self._model_dictionary = dict()
        self._scene_objects = list()

        self.SetCurrent(self._context)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(*self._background_color)
        glClearDepth(1.0)

    def add_scene_object(
        self,
        model_key: str,
        transform_to_world: Matrix4x4
    ) -> None:
        if model_key not in self._model_dictionary:
            raise RuntimeError(f"{model_key} is not a model that has been loaded into the current context.")
        model: Model = self._model_dictionary[model_key]
        self._scene_objects.append(GraphicsRenderer.SceneObject(model, transform_to_world))

    def clear_scene_objects(self):
        self._scene_objects.clear()

    def load_models_into_context_from_data_path(self) -> dict[str, Model]:
        # We may eventually want to consider letting the user define the path(s) from which to load data
        current_path: str = os.path.realpath(__file__)
        root_path: str = os.path.realpath(os.path.join(current_path, "..", "..", "..", "..", ".."))

        shaders: dict[str, Shader] = dict()
        shaders_path: str = os.path.join(root_path, "data", "graphics", "shaders")
        shader_code_path: str = os.path.realpath(os.path.join(current_path, "..", "..", "..", "graphics", "shaders"))
        for shader_filename in os.listdir(shaders_path):
            shader_filepath = os.path.join(shaders_path, shader_filename)
            shader_io: FileIO.Shader
            with open(shader_filepath, 'r') as f:
                shader_io = FileIO.Shader(**hjson.load(f))
            vertex_shader_code_path = os.path.join(shader_code_path, f"{shader_io.vertex_shader_label}.vert")
            vertex_shader_code: str
            with open(vertex_shader_code_path, 'r') as f:
                vertex_shader_code = f.read()
            fragment_shader_filepath = os.path.join(shader_code_path, f"{shader_io.vertex_shader_label}.frag")
            fragment_shader_code: str
            with open(fragment_shader_filepath, 'r') as f:
                fragment_shader_code = f.read()
            shader = Shader(
                vertex_shader_code=vertex_shader_code,
                fragment_shader_code=fragment_shader_code,
                property_list=shader_io.property_list)
            shader_basename = os.path.splitext(shader_filename)[0]
            shaders[shader_basename] = shader

        materials: dict[str, Material] = dict()
        materials_path: str = os.path.join(root_path, "data", "graphics", "materials")
        for material_filename in os.listdir(materials_path):
            material_filepath = os.path.join(materials_path, material_filename)
            material_io: FileIO.Material
            with open(material_filepath, 'r') as f:
                material_io = FileIO.Material(**hjson.load(f))
            material_basename = os.path.splitext(material_filename)[0]
            materials[material_basename] = Material(
                shader=shaders[material_io.shader_label],
                properties=material_io.properties)

        models_path: str = os.path.join(root_path, "data", "graphics", "models")
        geometry_path: str = os.path.join(root_path, "data", "graphics", "geometry")
        for model_filename in os.listdir(models_path):
            model_filepath = os.path.join(models_path, model_filename)
            model_io: FileIO.Model
            with open(model_filepath, 'r') as f:
                model_io = FileIO.Model(**hjson.load(f))
            model_parts: list[Model.ModelPart] = list()
            for model_part_io in model_io.parts:
                geometry_filename = f"{model_part_io.geometry_label}.stl"
                geometry_filepath = os.path.join(geometry_path, geometry_filename)
                geometry = stl.mesh.Mesh.from_file(geometry_filepath)
                vertices: list[numpy.array] = list()
                normals: list[numpy.array] = list()
                triangles: list[list[int]] = list()
                for i, face in enumerate(geometry.vectors):
                    for vertex in face:
                        vertices.append(vertex)
                    normals.append(geometry.normals[i])
                    triangles.append([i*3, i*3+1, i*3+2])
                material: Material = materials[model_part_io.material_label]
                model_part = Model.ModelPart(vertices, normals, triangles, material)
                model_parts.append(model_part)
            model_basename = os.path.splitext(model_filename)[0]
            self._model_dictionary[model_basename] = Model(parts=model_parts)

        self.SetCurrent(self._context)
        for key, model in self._model_dictionary.items():
            model.load_into_opengl()
            self._shaders = self._shaders.union(model.get_shaders())
        self._update_world_to_view()
        self._update_viewport()
        return self._model_dictionary

    def render(self):
        self.SetCurrent(self._context)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for scene_object in self._scene_objects:
            translation = [scene_object.transform_to_world[i, 3] for i in range(3)]
            rotation_matrx: numpy.ndarray = scene_object.transform_to_world.as_numpy_array()[0:3, 0:3]
            rotation_quaternion: numpy.ndarray = Rotation.from_matrix(rotation_matrx).as_quat(canonical=True)
            rotation_quaternion = rotation_quaternion.flatten()
            scene_object.model.draw(translation=translation, rotation_quaternion=list(rotation_quaternion))

        self.SwapBuffers()
        self._last_render_datetime_utc = datetime.datetime.utcnow()

    def _compute_seconds_since_last_update(self) -> float:
        return (datetime.datetime.utcnow() - self._last_update_datetime_utc).total_seconds()

    @staticmethod
    def _compute_mouse_deltas(
        current_mouse_pixels: float,  # along some dimension
        last_mouse_pixels: float,
        delta_time_seconds: float
    ) -> float:  # factor of maximum between -1 and 1
        delta_mouse_pixels: float = current_mouse_pixels - last_mouse_pixels
        velocity_mouse_pixels_per_second: float = delta_mouse_pixels / delta_time_seconds
        velocity_mouse_pixels_per_second = float(numpy.clip(
            a=velocity_mouse_pixels_per_second,
            a_min=-_INPUT_MAXIMUM_MOUSE_SPEED_PIXELS_PER_SECOND,
            a_max=_INPUT_MAXIMUM_MOUSE_SPEED_PIXELS_PER_SECOND))
        # Currently linear, but maybe we try some other function at some point
        factor: float = velocity_mouse_pixels_per_second / _INPUT_MAXIMUM_MOUSE_SPEED_PIXELS_PER_SECOND
        return factor

    def _on_mouse_button_left_down(self, _evt: wx.MouseEvent):
        self._mouse_button_left_is_down = True

    def _on_mouse_button_left_up(self, _evt: wx.MouseEvent):
        self._mouse_button_left_is_down = False

    def _on_mouse_button_middle_down(self, _evt: wx.MouseEvent):
        self._mouse_button_middle_is_down = True

    def _on_mouse_button_middle_up(self, _evt: wx.MouseEvent):
        self._mouse_button_middle_is_down = False

    def _on_mouse_button_right_down(self, _evt: wx.MouseEvent):
        self._mouse_button_right_is_down = True

    def _on_mouse_button_right_up(self, _evt: wx.MouseEvent):
        self._mouse_button_right_is_down = False

    def _on_mouse_moved(self, event: wx.MouseEvent):
        delta_time_seconds: float = self._compute_seconds_since_last_update()
        if delta_time_seconds <= 0:
            return  # appears to happen somewhat frequently in Windows, perhaps multiple events handled back-to-back?

        self._last_mouse_x_pixels = self._current_mouse_x_pixels
        self._current_mouse_x_pixels = event.GetPosition().x
        delta_x_factor = self._compute_mouse_deltas(
            current_mouse_pixels=self._current_mouse_x_pixels,
            last_mouse_pixels=self._last_mouse_x_pixels,
            delta_time_seconds=delta_time_seconds)
        self._last_mouse_y_pixels = self._current_mouse_y_pixels
        self._current_mouse_y_pixels = event.GetPosition().y
        delta_y_factor = self._compute_mouse_deltas(
            current_mouse_pixels=self._current_mouse_y_pixels,
            last_mouse_pixels=self._last_mouse_y_pixels,
            delta_time_seconds=delta_time_seconds)

        if self._mouse_button_left_is_down:
            delta_x_degrees: float = \
                delta_x_factor * _INPUT_MAXIMUM_ROTATION_SPEED_DEGREES_PER_SECOND * delta_time_seconds
            self._perspective_azimuth_degrees += delta_x_degrees
            while self._perspective_azimuth_degrees < _PERSPECTIVE_AZIMUTH_DEGREES_MINIMUM:
                self._perspective_azimuth_degrees += 360.0
            while self._perspective_azimuth_degrees > _PERSPECTIVE_AZIMUTH_DEGREES_MAXIMUM:
                self._perspective_azimuth_degrees -= 360.0
            delta_y_degrees: float = \
                delta_y_factor * _INPUT_MAXIMUM_ROTATION_SPEED_DEGREES_PER_SECOND * delta_time_seconds
            self._perspective_altitude_degrees += delta_y_degrees
            self._perspective_altitude_degrees = float(numpy.clip(
                a=self._perspective_altitude_degrees,
                a_min=_PERSPECTIVE_ALTITUDE_DEGREES_MINIMUM,
                a_max=_PERSPECTIVE_ALTITUDE_DEGREES_MAXIMUM))
            self._update_world_to_view()

        if self._mouse_button_right_is_down:
            delta_fov_degrees = delta_y_factor * _INPUT_MAXIMUM_ZOOM_SPEED_DEGREES_PER_SECOND
            self._fov_degrees += delta_fov_degrees
            self._fov_degrees = float(numpy.clip(
                a=self._fov_degrees,
                a_min=_VIEWPORT_FOV_DEGREES_MINIMUM,
                a_max=_VIEWPORT_FOV_DEGREES_MAXIMUM))
            self._update_viewport()

        if self._mouse_button_middle_is_down:
            delta_x_millimeters = delta_x_factor * _INPUT_MAXIMUM_LATERAL_SPEED_MILLIMETERS_PER_SECOND
            translation_x_vector = self._current_view_to_world_matrix[0:3, 0]
            self._perspective_target = list(
                numpy.array(self._perspective_target) - delta_x_millimeters * translation_x_vector)
            delta_y_millimeters = delta_y_factor * _INPUT_MAXIMUM_LATERAL_SPEED_MILLIMETERS_PER_SECOND
            translation_y_vector = self._current_view_to_world_matrix[0:3, 1]
            self._perspective_target = list(
                numpy.array(self._perspective_target) + delta_y_millimeters * translation_y_vector)
            self._update_world_to_view()

        self._last_update_datetime_utc = datetime.datetime.utcnow()

    def _on_mouse_wheel(self, event: wx.MouseEvent):
        delta_time_seconds: float = self._compute_seconds_since_last_update()
        if delta_time_seconds <= 0:
            return  # appears to happen somewhat frequently in Windows, perhaps multiple events handled back-to-back?

        delta_wheel: int = event.GetWheelRotation()
        print(delta_wheel)
        velocity_units_per_second: float = delta_wheel / delta_time_seconds
        velocity_units_per_second = float(numpy.clip(
            a=velocity_units_per_second,
            a_min=-_INPUT_MAXIMUM_SCROLL_SPEED_UNITS_PER_SECOND,
            a_max=_INPUT_MAXIMUM_SCROLL_SPEED_UNITS_PER_SECOND))
        delta_z_factor: float = velocity_units_per_second / _INPUT_MAXIMUM_SCROLL_SPEED_UNITS_PER_SECOND
        delta_z_millimeters = delta_z_factor * _INPUT_MAXIMUM_DOLLY_SPEED_MILLIMETERS_PER_SECOND
        translation_z_vector = self._current_view_to_world_matrix[0:3, 2]
        self._perspective_target = list(
            numpy.array(self._perspective_target) - delta_z_millimeters * translation_z_vector)
        self._update_world_to_view()

        self._last_update_datetime_utc = datetime.datetime.utcnow()

    def _on_resize(self, _evt: wx.SizeEvent):
        self._update_viewport()
        self.Refresh()

    def _update_viewport(self):
        self.SetCurrent(self._context)
        width_px: float = self.GetSize()[0]
        height_px: float = self.GetSize()[1]
        glViewport(0, 0, width_px, height_px)
        for shader in self._shaders:
            shader_program = shader.get_shader_program()
            glUseProgram(shader_program)
            view_to_clip_matrix = numpy.zeros((4, 4), dtype="float32")
            fov_radians = self._fov_degrees * numpy.pi / 180.0
            aspect_ratio = width_px / height_px
            near = 1.0  # 1 millimeter
            far = 1000.0  # 1 meter
            right = numpy.tan(fov_radians/2.0) * near
            left = -right
            top = right / aspect_ratio
            bottom = -top
            view_to_clip_matrix[0, 0] = (2 * near) / (right - left)
            view_to_clip_matrix[0, 2] = (right + left) / (right - left)
            view_to_clip_matrix[1, 1] = (2 * near) / (top - bottom)
            view_to_clip_matrix[1, 2] = (top + bottom) / (top - bottom)
            view_to_clip_matrix[2, 2] = -((far + near) / (far - near))
            view_to_clip_matrix[2, 3] = -(2 * near * far) / (far - near)
            view_to_clip_matrix[3, 2] = -1
            view_to_clip_location = glGetUniformLocation(shader_program, Constants.GL_VIEW_TO_CLIP_PROPERTY_KEY)
            glUniformMatrix4fv(view_to_clip_location, 1, Constants.GL_TRANSPOSE_MATRIX, view_to_clip_matrix)

    def _update_world_to_view(self):
        self.SetCurrent(self._context)
        for shader in self._shaders:
            shader_program = shader.get_shader_program()
            glUseProgram(shader_program)
            # if we use zero as an euler angle, then we look down on scene.
            # What we want is to be in line with the equator i.e. xy-plane
            # then rotate sideways and up or down
            adjusted_altitude = self._perspective_altitude_degrees - 90.0
            eye_rotation_matrix = Rotation.from_euler(
                seq="ZX",
                angles=[-self._perspective_azimuth_degrees, -adjusted_altitude],
                degrees=True).as_matrix()
            eye_position = \
                numpy.matmul(eye_rotation_matrix, [0.0, 0.0, self._perspective_distance]) + \
                numpy.array(self._perspective_target)
            self._current_view_to_world_matrix[0:3, 0:3] = eye_rotation_matrix
            self._current_view_to_world_matrix[0:3, 3] = eye_position
            world_to_view_matrix = numpy.linalg.inv(self._current_view_to_world_matrix)
            world_to_view_location = glGetUniformLocation(shader_program, Constants.GL_WORLD_TO_VIEW_PROPERTY_KEY)
            glUniformMatrix4fv(world_to_view_location, 1, Constants.GL_TRANSPOSE_MATRIX, world_to_view_matrix)
