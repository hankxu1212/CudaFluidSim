#include "SpatialHashTable.hpp"

// implementation based on
// https://sph-tutorial.physics-simulation.org/slides/01_intro_foundations_neighborhood.pdf
uint32_t SpatialHashTable::GetHash(const glm::ivec2& cell)
{
    return (
        static_cast<uint32_t>(cell.x * 73856093)
        ^ static_cast<uint32_t>(cell.y * 19349663)
    ) % TABLE_SIZE;
}

glm::ivec2 SpatialHashTable::GetCell(const Particle& p, float h)
{
    assert(h > 0);
    glm::vec2 clampedPos = glm::clamp(p.position, glm::vec2(0), p.position);

    return { std::floor(clampedPos.x / h), std::floor(clampedPos.y / h)};
}

uint32_t SpatialHashTable::GetHashFromParticle(const Particle& p, float h)
{
    return GetHash(GetCell(p, h));
}

std::vector<uint32_t> SpatialHashTable::Create(const std::vector<Particle>& sortedParticles)
{
    std::vector<uint32_t> particleTable(TABLE_SIZE, NO_PARTICLE);

    uint32_t prevHash = NO_PARTICLE;
    for (size_t i = 0; i < sortedParticles.size(); ++i) 
    {
        uint32_t currentHash = sortedParticles[i].hash;
        if (currentHash != prevHash) {
            particleTable[currentHash] = i;
            prevHash = currentHash;
        }
    }
    return particleTable;
}

void SpatialHashTable::CreateNonAlloc(const std::vector<Particle>& sortedParticles, std::vector<uint32_t>& out)
{
    memset(out.data(), NO_PARTICLE, out.size() * sizeof(uint32_t));

    uint32_t prevHash = NO_PARTICLE;
    for (size_t i = 0; i < sortedParticles.size(); ++i)
    {
        uint32_t currentHash = sortedParticles[i].hash;
        if (currentHash != prevHash) {
            out[currentHash] = i;
            prevHash = currentHash;
        }
    }
}
