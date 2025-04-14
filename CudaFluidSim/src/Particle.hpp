#pragma once

#include "glm/glm.hpp"
#include "SolverConfiguration.hpp"

struct alignas(64) Particle {  // Match cache line size
	glm::vec2 position;    // 8 bytes
	glm::vec2 velocity;    // 8 bytes
	glm::vec2 force;       // 8 bytes
	float density;         // 4 bytes
	float pressure;        // 4 bytes
	uint32_t hash;         // 4 bytes

	Particle() :
		position(0, 0), velocity(0.f, 0.f), force(0.f, 0.f), density(0), pressure(0.f) {
	}

	Particle(float _x, float _y)
		: position(_x, _y), velocity(0.f, 0.f), force(0.f, 0.f), density(0), pressure(0.f) {
	}
};