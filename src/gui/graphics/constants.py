from typing import Final
from OpenGL.GL import *


class Constants:

    GL_TRANSPOSE_MATRIX: Final[GLboolean] = GL_TRUE

    # These must be specified to all shaders
    GL_VIEW_TO_CLIP_PROPERTY_KEY: Final[str] = "a_viewToClipTransform"
    GL_OBJECT_TO_WORLD_PROPERTY_KEY: Final[str] = "a_objectToWorldTransform"
    GL_WORLD_TO_VIEW_PROPERTY_KEY: Final[str] = "a_worldToViewTransform"
