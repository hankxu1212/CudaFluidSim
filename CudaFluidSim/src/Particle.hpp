#pragma once

#include <vector_functions.h>

#include "glm/glm.hpp"
#include "SolverConfiguration.hpp"

struct alignas(32) d_Particle {
	float4 position_density_pressure; // 16 bytes
	float4 velocity_force; // 16 bytes

	d_Particle() : position_density_pressure(make_float4(0.f, 0.f, 0.f, 0.f)), velocity_force(make_float4(0.f, 0.f, 0.f, 0.f)) {}

	d_Particle(float _x, float _y) : position_density_pressure(make_float4(_x, _y, 0.f, 0.f)), velocity_force(make_float4(0.f, 0.f, 0.f, 0.f)) {}
};

struct alignas(64) Particle {  // Match cache line size
	glm::vec2 position;    // 8 bytes
	glm::vec2 velocity;    // 8 bytes
	glm::vec2 velocityHalf; // 8 bytes
	glm::vec2 force;       // 8 bytes
	float density;         // 4 bytes
	float pressure;        // 4 bytes
	uint32_t hash;         // 4 bytes

	Particle() :
		position(0, 0), velocity(0.f, 0.f), velocityHalf(0,0), force(0.f, 0.f), density(0), pressure(0.f) {
	}

	Particle(float _x, float _y)
		: position(_x, _y), velocity(0.f, 0.f), velocityHalf(0, 0), force(0.f, 0.f), density(0), pressure(0.f) {
	}

	Particle(d_Particle p) :
		position(p.position_density_pressure.x, p.position_density_pressure.y), velocity(p.velocity_force.x, p.velocity_force.y), velocityHalf(0, 0), force(p.velocity_force.w, p.velocity_force.z), density(p.position_density_pressure.z), pressure(p.position_density_pressure.w) {
	}
};
