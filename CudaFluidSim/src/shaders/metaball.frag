#version 430 core
in vec2 TexCoords;
out vec4 FragColor;

// SSBO for metaball positions – binding 0.
layout(std430, binding = 0) buffer Metaballs {
    vec2 metaballs[];
};
// SSBO for per-cell counts – binding 1.
layout(std430, binding = 1) buffer CellCounts {
    int cellCounts[];
};
// SSBO for per-cell metaball indices – binding 2.
layout(std430, binding = 2) buffer CellIndices {
    int cellIndices[];
};

uniform float metaballRadius;
uniform float threshold;
uniform int tileSize;
uniform int gridWidth;
uniform int gridHeight;  // Added so we know the vertical grid dimension.
const int maxIndicesPerCell = 64; // Must match the CPU-side constant.

void main(){
    vec2 fragCoord = gl_FragCoord.xy;
    
    // Determine the current cell for this fragment.
    int cellX = int(fragCoord.x) / tileSize;
    int cellY = int(fragCoord.y) / tileSize;

    float field = 0.0;
    // Loop over the current cell and its 8 neighbors.
    for (int offsetY = -1; offsetY <= 1; offsetY++){
        for (int offsetX = -1; offsetX <= 1; offsetX++){
            // Clamp neighbor indices to valid grid ranges.
            int neighborX = clamp(cellX + offsetX, 0, gridWidth - 1);
            int neighborY = clamp(cellY + offsetY, 0, gridHeight - 1);
            int cellID = neighborY * gridWidth + neighborX;
            
            int count = cellCounts[cellID];
            for (int j = 0; j < count; j++){
                int metaballIndex = cellIndices[cellID * maxIndicesPerCell + j];
                vec2 pos = metaballs[metaballIndex];
                float dx = fragCoord.x - pos.x;
                float dy = fragCoord.y - pos.y;
                float d2 = dx * dx + dy * dy;
                field += exp(-d2 / (metaballRadius * metaballRadius));
            }
        }
    }
    
    // If the field is below threshold, output background.
    if (field < threshold) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Map the field to a normalized parameter t.
    float t = clamp((field - threshold) / threshold, 0.0, 1.0);
    
    // Define a cute, pastel color gradient.
    vec3 colorLow   = vec3(1.0, 0.8, 0.9); // Soft pastel pink.
    vec3 colorMid   = vec3(0.9, 0.85, 1.0); // Pastel lavender.
    vec3 colorHigh  = vec3(0.8, 1.0, 1.0);  // Pastel cyan.
    
    vec3 fluidColor;
    if (t < 0.5) {
        fluidColor = mix(colorLow, colorMid, t * 2.0);
    } else {
        fluidColor = mix(colorMid, colorHigh, (t - 0.5) * 2.0);
    }
    
    FragColor = vec4(fluidColor, 1.0);
}
