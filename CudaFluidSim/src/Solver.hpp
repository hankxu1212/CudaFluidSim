#pragma once

#include "window/Window.hpp"
#include "glm/glm.hpp"

class Solver : public Module::Registrar<Solver>
{
	inline static const bool Registered = Register(
		UpdateStage::Pre, 
		DestroyStage::Normal,
		Requires<Window>()
	);

public:
	Solver();

	virtual ~Solver();

	void Update();

	struct Particle
	{
		Particle(float _x, float _y)
			: x(_x, _y), v(0.f, 0.f), f(0.f, 0.f), rho(0), p(0.f) {}

		glm::vec2 x, v, f;
		float rho, p;
	};

	std::vector<Particle> m_Particles;

	void InitSPH();

	void Integrate(float dt);

	void ComputeDensityPressure();

	void ComputeForces();
};

