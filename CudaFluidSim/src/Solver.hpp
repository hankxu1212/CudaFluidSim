#pragma once

#include <array>

#include "window/Window.hpp"
#include "Particle.hpp"
#include "events/KeyEvent.hpp"
#include "SolverConfiguration.hpp"
#include "SpatialHashTable.hpp"

class Solver : public Layer
{
public:
	static Solver* Get() { return s_Instance; }
	static Solver* s_Instance;

	std::array<Particle, NUM_PARTICLES> m_Particles;
	std::array<d_Particle, NUM_PARTICLES> m_dParticles;

	uint32_t numThreads;

	Solver();

	virtual ~Solver();

	void OnUpdate() override;

	void OnEvent(Event& event) override;

	bool OnKeyPressed(KeyPressedEvent& e);

	void OnImGuiRender() override;

	bool Paused = false;
	bool Restart = false;
	float LastFrameUpdateTime;

private:

	void OnRestart();

	// forward euler integration with fixed delta time
	void Integrate(float dt);

	void LeapfrogDrift(float dt);

	void LeapfrogKick(float dt);

#pragma region Basic OpenMP

	void InitSPH();

	void ComputeDensityPressure();

	void ComputeForces();

#pragma endregion

#pragma region Parallel with Spatial Hashing

	void SpatialParallelInitSPH();

	void SpatialParallelComputeDensityPressure();

	void SpatialParallelComputeForces();

	void SpatialParallelComputeCombined();

	void SpatialParallelUpdate();

	// should rarely be used since vector emplaces are expensive
	std::vector<uint32_t> FindNearbyParticles(int sortedPid);

	std::array<uint32_t, SpatialHashTable::TABLE_SIZE> m_ParticleHashTable;

#pragma endregion
};

