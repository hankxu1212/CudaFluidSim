#version 330 core
out vec4 FragColor;
in vec2 TexCoords;
uniform sampler2D scene;
uniform float blurSize; // Adjusts blur spread.
void main() {
    vec2 tex_offset = 1.0 / vec2(800.0, 600.0); // for 800x600 resolution
    vec3 result = texture(scene, TexCoords).rgb * 0.2270270270;
    result += texture(scene, TexCoords + vec2(tex_offset.x * blurSize, 0.0)).rgb * 0.3162162162;
    result += texture(scene, TexCoords - vec2(tex_offset.x * blurSize, 0.0)).rgb * 0.3162162162;
    result += texture(scene, TexCoords + vec2(0.0, tex_offset.y * blurSize)).rgb * 0.0702702703;
    result += texture(scene, TexCoords - vec2(0.0, tex_offset.y * blurSize)).rgb * 0.0702702703;
    FragColor = vec4(result, 1.0);
}