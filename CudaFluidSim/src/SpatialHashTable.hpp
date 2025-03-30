#pragma once

#include "Particle.hpp"


class SpatialHashTable
{
public:
    const static uint32_t TABLE_SIZE = 262144;
    const static uint32_t NO_PARTICLE = 0xFFFFFFFF;

    // Returns a hash of the cell position
    static uint32_t GetHash(const glm::ivec2& cell);

    // Get the cell that the particle is in.
    static glm::ivec2 GetCell(const Particle& p, float h);

    static uint32_t GetHashFromParticle(const Particle& p, float h);

    // Creates the particle neighbor hash table.
    static std::vector<uint32_t> Create(const std::vector<Particle>& sortedParticles);
};
