#pragma once

#include "glm/glm.hpp"

struct Particle
{
	Particle() :
		position(0, 0), velocity(0.f, 0.f), force(0.f, 0.f), density(0), pressure(0.f), id(0), debugColor(0) {}

	Particle(float _x, float _y, uint16_t _id, uint16_t _colorId)
		: position(_x, _y), velocity(0.f, 0.f), force(0.f, 0.f), density(0), pressure(0.f), id(_id), debugColor(_colorId) {}

	glm::vec2 position, velocity, force;
	float density, pressure;

	// spatial hashing
	uint32_t hash;

	uint16_t id; // spawning id [0, NUM_PARTICLES]
	float debugColor; // debug color index
};