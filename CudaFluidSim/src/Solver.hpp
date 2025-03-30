#pragma once

#include "window/Window.hpp"
#include "Particle.hpp"
#include "events/KeyEvent.hpp"

class Solver : public Layer
{
public:
	static Solver* Get() { return s_Instance; }
	static Solver* s_Instance;

	// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.

	// solver parameters
	constexpr static int H = 12;				// kernel radius
	constexpr static int CELL_SIZE = H * 2;		// spatial grid size
	constexpr static float HSQ = H * H;		   // radius^2 for optimization
	constexpr static float REST_DENS = 300.f;  // rest density
	constexpr static float GAS_CONST = 2000.f; // const for equation of state
	constexpr static float MASS = 2.5f;		   // assume all particles have the same mass
	constexpr static float VISC = 2000.f;	   // viscosity constant
	constexpr static float DT = 0.0005f;       // simulation delta time

	// smoothing kernels defined in Müller and their gradients
	// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
	float POLY6;
	float SPIKY_GRAD;
	float VISC_LAP;

	// interaction
	const static int NUM_PARTICLES = 2000;

	// rendering projection parameters

	// simulation parameters
	constexpr static float EPS = H; // boundary epsilon
	constexpr static float BOUND_DAMPING = -0.1f;
	
	std::vector<Particle> m_Particles;

	Solver();

	virtual ~Solver();

	void OnUpdate() override;

	void OnEvent(Event& event) override;

	bool OnKeyPressed(KeyPressedEvent& e);

private:

	bool isPaused = false;

	// forward euler integration with fixed delta time
	void Integrate(float dt);

#pragma region Basic OpenMP

	void InitSPH();

	void ComputeDensityPressure();

	void ComputeForces();

#pragma endregion

#pragma region Parallel with Spatial Hashing

	void SpatialParallelInitSPH();

	void SpatialParallelComputeDensityPressure();

	void SpatialParallelComputeForces();

	void CalculateHashes();

	void SpatialParallelUpdate();

	std::vector<uint32_t> FindNearbyParticles(int sortedPid);

	std::vector<uint32_t> m_ParticleHashTable;

#pragma endregion

#pragma region CUDA

	// TODO: add your CUDA variables/kernel functions here
	// these are just placeholder empty functions for now

	void KernelInitSPH();

	void KernelComputeDensityPressure();

	void KernelComputeForces();

	void KernelIntegrate(float dt);

#pragma endregion
};

