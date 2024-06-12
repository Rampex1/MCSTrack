#version 330 core

layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;

uniform mat4 a_objectToWorldTransform;
uniform mat4 a_worldToViewTransform;
uniform mat4 a_viewToClipTransform;

out vec3 b_normal;

void main()
{
    mat4 objectToViewTransform = a_worldToViewTransform * a_objectToWorldTransform;
    gl_Position = a_viewToClipTransform * objectToViewTransform * vec4(a_position.x, a_position.y, a_position.z, 1.0);
    vec4 normal = objectToViewTransform * vec4(a_normal, 0.0);
    b_normal = normal.xyz;
}
