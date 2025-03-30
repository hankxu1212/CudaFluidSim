#version 330 core
layout (location = 0) in vec4 vertex;

flat out vec3 particleColor;

uniform float particleIndex;
uniform mat4 projection;
uniform vec2 center;

float random(float seed) {
    return fract(sin(seed * 78.233) * 43758.5453);
}

void main()
{
    float scale = 8f;
    gl_Position = projection * vec4((vertex.xy * scale) + center, 0.0, 1.0);

    float r = random(particleIndex * 0.123);
    particleColor = vec3(r, r, r);
}