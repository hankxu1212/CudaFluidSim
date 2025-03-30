#version 330 core

out vec4 fragColor;
flat in vec3 particleColor;

void main()
{
    fragColor = vec4(particleColor, 1);
}