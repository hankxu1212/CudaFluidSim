#version 330 core
layout (location = 0) in vec4 vertex;

flat out vec3 particleColor;

uniform float particleRGB;
uniform mat4 projection;
uniform vec2 center;

void main()
{
    float scale = 8f;
    gl_Position = projection * vec4((vertex.xy * scale) + center, 0.0, 1.0);

    particleColor = vec3(particleRGB, particleRGB, particleRGB);
}