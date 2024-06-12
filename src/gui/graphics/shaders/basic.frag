#version 330 core

in vec3 b_normal;
out vec4 c_fragColor;

uniform vec3 b_ambientColor;
uniform vec3 b_diffuseColor;
uniform float b_opacity;
uniform vec3 b_lightDirectionView;

void main()
{
    float diffuseFactor = clamp(dot(normalize(-b_lightDirectionView), normalize(b_normal)), 0.0, 1.0);
    vec3 diffuseContribution = diffuseFactor * b_diffuseColor;
    vec3 color = clamp((b_ambientColor + diffuseContribution), 0.0, 1.0);
    c_fragColor = vec4(color, b_opacity);
}
