#include "SpatialHashTable.hpp"
#include "morton.h"

// implementation based on
// https://sph-tutorial.physics-simulation.org/slides/01_intro_foundations_neighborhood.pdf
uint32_t SpatialHashTable::GetHash(const glm::ivec2& cell)
{
    return (
        static_cast<uint32_t>(cell.x * 73856093)
        ^ static_cast<uint32_t>(cell.y * 19349663)
    ) % TABLE_SIZE;
}

uint32_t SpatialHashTable::GetZOrderHash(const glm::ivec2& cell)
{
    return morton2D_32_encode(
        static_cast<uint_fast32_t>(cell[0] -
            (std::numeric_limits<int>::lowest() + 1)),
        static_cast<uint_fast32_t>(cell[1] -
            (std::numeric_limits<int>::lowest() + 1)));
}

glm::ivec2 SpatialHashTable::GetCell(const Particle& p, float h)
{
    assert(h > 0);
    glm::vec2 clampedPos = glm::clamp(p.position, glm::vec2(0), p.position);

    return { std::floor(clampedPos.x / h), std::floor(clampedPos.y / h)};
}

uint32_t SpatialHashTable::GetHashFromParticle(const Particle& p, float h, bool useZOrder)
{
    //if (useZOrder)
    //    return GetZOrderHash(GetCell(p, h));
    //else
        return GetHash(GetCell(p, h));
}

std::vector<uint32_t> SpatialHashTable::Create(const Particle* particles, uint32_t count)
{
    std::vector<uint32_t> particleTable(TABLE_SIZE, NO_PARTICLE);

    uint32_t prevHash = NO_PARTICLE;
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t currentHash = particles[i].hash;
        if (currentHash != prevHash) {
            particleTable[currentHash] = i;
            prevHash = currentHash;
        }
    }
    return particleTable;
}

void SpatialHashTable::CreateNonAlloc(const Particle* particles, uint32_t count, uint32_t* out)
{
    memset(out, NO_PARTICLE, TABLE_SIZE * sizeof(uint32_t));

    uint32_t prevHash = NO_PARTICLE;
    for (size_t i = 0; i < count; ++i)
    {
        uint32_t currentHash = particles[i].hash;
        if (currentHash != prevHash) {
            out[currentHash] = i;
            prevHash = currentHash;
        }
    }
}
