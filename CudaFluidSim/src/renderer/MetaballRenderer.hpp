#pragma once

#include "resources/ResourceManager.h"
#include "math/Math.hpp"
#include "resources/Files.hpp"
#include "Solver.hpp"

//--------------------------------------------------------------
// MetaballRenderer: Renders a full-screen quad and computes the scalar field.
//--------------------------------------------------------------
class MetaballRenderer {
public:
    // Constructor: screen dimensions are needed to compute grid dimensions.
    MetaballRenderer(int screenWidth, int screenHeight)
        : quadVAO(0), quadVBO(0),
        ssboMetaballs(0), ssboCellCounts(0), ssboCellIndices(0),
        tileSize(32), screenWidth(screenWidth), screenHeight(screenHeight)
    {
        gridWidth = (screenWidth + tileSize - 1) / tileSize;
        gridHeight = (screenHeight + tileSize - 1) / tileSize;
        Initialize();
    }

    ~MetaballRenderer() {
        Cleanup();
    }

public:

    float MetaballThreshold = 0.5f;

    void Initialize() {
        const std::string pathRelativeToExecutable = "../../CudaFluidSim/src/";
        std::string vertPath = Files::Path(pathRelativeToExecutable + "shaders/quad.vert");
        std::string fragPath = Files::Path(pathRelativeToExecutable + "shaders/metaball.frag");
        shader = ResourceManager::LoadShader(vertPath.c_str(), fragPath.c_str(), nullptr, "metaball");

        // Set up full-screen quad (positions and texture coordinates).
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,   0.0f, 1.0f,
            -1.0f, -1.0f,   0.0f, 0.0f,
             1.0f, -1.0f,   1.0f, 0.0f,

            -1.0f,  1.0f,   0.0f, 1.0f,
             1.0f, -1.0f,   1.0f, 0.0f,
             1.0f,  1.0f,   1.0f, 1.0f
        };

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // Create SSBO for metaball positions.
        const int maxParticles = 50000; // Adjust as needed.
        glGenBuffers(1, &ssboMetaballs);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMetaballs);
        glBufferData(GL_SHADER_STORAGE_BUFFER, maxParticles * sizeof(glm::vec2), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMetaballs);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Create SSBO for cell counts (gridWidth * gridHeight integers).
        glGenBuffers(1, &ssboCellCounts);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCellCounts);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridWidth * gridHeight * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboCellCounts);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // Create SSBO for cell indices (gridWidth * gridHeight * maxIndicesPerCell integers).
        constexpr int maxIndicesPerCell = 64;
        glGenBuffers(1, &ssboCellIndices);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCellIndices);
        glBufferData(GL_SHADER_STORAGE_BUFFER, gridWidth * gridHeight * maxIndicesPerCell * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboCellIndices);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    glm::vec2 TransformToScreenSpace(const glm::vec2& pos, const glm::mat4& camMatrix, int screenWidth, int screenHeight)
    {
        // Create a homogeneous coordinate with z=0 and w=1.
        glm::vec4 clip = camMatrix * glm::vec4(pos, 0.0f, 1.0f);

        // Perspective divide (for orthographic, clip.w is normally 1, but it’s a good habit)
        glm::vec3 ndc = glm::vec3(clip) / clip.w;

        // Map normalized device coordinates to screen space.
        float sx = (ndc.x * 0.5f + 0.5f) * screenWidth;
        float sy = (ndc.y * 0.5f + 0.5f) * screenHeight;

        return glm::vec2(sx, sy);
    }

    // Uploads the positions (from particles) into the SSBO.
    // Note: Because Particle::position is not tightly packed in the Particle struct,
    // we extract positions into a temporary std::vector<glm::vec2>.
    void UpdateSSBO(const Particle* particles, int count) {
        std::vector<glm::vec2> positions(count);
        static glm::mat4 cameraMatrix = glm::ortho(
            0.0f,
            static_cast<float>(Window::Get()->m_Data.Width),
            static_cast<float>(Window::Get()->m_Data.Height),
            0.0f,
            -1.0f,
            1.0f
        );

        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            positions[i] = TransformToScreenSpace(
                particles[i].position,
                cameraMatrix,   // e.g., projection * view (or just projection if view is identity)
                Window::Get()->m_Data.Width,
                Window::Get()->m_Data.Height
            );
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMetaballs);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * sizeof(glm::vec2), positions.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        // CPU-side function to build spatial partitioning.
        // Temporary arrays for cell counts and indices.
        std::vector<int> cellCounts(gridWidth * gridHeight, 0);
        std::vector<int> cellIndices(gridWidth * gridHeight * 64, -1); // 64 == maxIndicesPerCell.

        #pragma omp parallel for
        for (int i = 0; i < positions.size(); ++i) {
            const glm::vec2& pos = positions[i];
            int cellX = std::min((int)(pos.x / tileSize), gridWidth - 1);
            int cellY = std::min((int)(pos.y / tileSize), gridHeight - 1);
            int cellID = cellY * gridWidth + cellX;
            int indexInCell = cellCounts[cellID];
            
            #pragma omp critical
            {
                if (indexInCell < 64) {
                    cellIndices[cellID * 64 + indexInCell] = i;
                    cellCounts[cellID]++;
                }
            }
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCellCounts);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cellCounts.size() * sizeof(int), cellCounts.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboCellIndices);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, cellIndices.size() * sizeof(int), cellIndices.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // Render the metaballs given the current particle positions.
    // Only the positions of the first 'metaballCount' particles are used.
    void Render(const Particle* particles, int count) {
        // Update the SSBO with current particle positions.
        UpdateSSBO(particles, count);

        shader.Use();
        shader.SetInteger("metaballCount", count);
        shader.SetFloat("metaballRadius", H); // Adjust radius as needed.
        shader.SetFloat("threshold", MetaballThreshold);          // Adjust threshold for fluid surface.
        shader.SetInteger("tileSize", tileSize);
        shader.SetInteger("gridWidth", gridWidth);
        shader.SetInteger("gridHeight", gridHeight);     // Pass gridHeight to the shader.

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    }

    void Cleanup() {
        if (quadVBO) { glDeleteBuffers(1, &quadVBO); quadVBO = 0; }
        if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
        if (ssboMetaballs) { glDeleteBuffers(1, &ssboMetaballs); ssboMetaballs = 0; }
        if (ssboCellCounts) { glDeleteBuffers(1, &ssboCellCounts); ssboCellCounts = 0; }
        if (ssboCellIndices) { glDeleteBuffers(1, &ssboCellIndices); ssboCellIndices = 0; }
        glDeleteProgram(shader.ID);
    }

private:
    unsigned int quadVAO, quadVBO;
    unsigned int ssboMetaballs, ssboCellCounts, ssboCellIndices;
    Shader shader;

    // Configuration and grid dimensions.
    int tileSize;
    int screenWidth, screenHeight;
    int gridWidth, gridHeight;
};
