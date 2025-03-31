#include "Solver.hpp"
#include "math/Math.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

#include <omp.h>

#include "utils/Time.hpp"
#include "utils/Timer.hpp"

#include "SpatialHashTable.hpp"

#define WINDOW_HEIGHT Window::Get()->m_Data.Height
#define WINDOW_WIDTH Window::Get()->m_Data.Width

const static glm::vec2 G(0.f, 10.f);   // external (gravitational) forces

//#define PROFILE_TIMES
#define PROFILE_FRAME_TIME

Solver* Solver::s_Instance = nullptr;

Solver::Solver()
{
	s_Instance = this;

	POLY6 = 4.f / (glm::pi<float>() * pow(H, 8.f));
	SPIKY_GRAD = -10.f / (glm::pi<float>() * pow(H, 5.f));
	VISC_LAP = 40.f / (glm::pi<float>() * pow(H, 5.f));

	numThreads = Application::GetSpecification().numThreads;
	std::cout << "Solving with " << numThreads << " OpenMP threads.\n";

	omp_set_num_threads(numThreads);

	auto accelerationMode = Application::GetSpecification().accelerationMode;

	switch (accelerationMode)
	{
	case ApplicationSpecification::Naive:
		std::cout << "Acceleration method: Naive\n";
		InitSPH();
		break;

	case ApplicationSpecification::Spatial:
		std::cout << "Acceleration method: Spatial Hashing\n";
		SpatialParallelInitSPH();
		break;

	case ApplicationSpecification::GPU:
		std::cout << "Acceleration method: GPU\n";
		KernelInitSPH();
		break;
	}
}

Solver::~Solver()
{
}

void Solver::OnUpdate()
{
	if (isPaused)
		return;

	auto accelerationMode = Application::GetSpecification().accelerationMode;

	switch (accelerationMode)
	{
	case ApplicationSpecification::Naive:
		ComputeDensityPressure();
		ComputeForces();
		Integrate(DT);
		break;

	case ApplicationSpecification::Spatial:
		SpatialParallelUpdate();
		break;

	case ApplicationSpecification::GPU:
		// TODO: run your kernel code here
		KernelComputeDensityPressure();
		KernelComputeForces();
		KernelIntegrate(DT);
		break;
	}
}

void Solver::OnEvent(Event& e)
{
	e.Dispatch<KeyPressedEvent>(NE_BIND_EVENT_FN(Solver::OnKeyPressed));
}

bool Solver::OnKeyPressed(KeyPressedEvent& e)
{
	if (e.m_IsRepeat)
		return false;

	switch (e.m_KeyCode)
	{
	case Key::Escape:
		isPaused = !isPaused;
		break;
	}

	return false;
}

void Solver::InitSPH()
{
	for (int i = 0; i < NUM_PARTICLES; ++i) {
		// Calculate random angle
		float angle = Math::Random(0, 2.0f * 3.1415f);
		float r = Math::Random();

		// Calculate random position on circle
		float x = WINDOW_HEIGHT / 3 + WINDOW_HEIGHT / 3 * r * cos(angle) + WINDOW_HEIGHT / 5;
		float y = WINDOW_HEIGHT / 3 + WINDOW_HEIGHT / 3 * r * sin(angle);

		m_Particles.emplace_back(x, y, i, 0);
	}

	std::cout << "Initializing " << m_Particles.size() << " particles" << std::endl;
}

void Solver::Integrate(float dt)
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		auto& p = m_Particles[i];

		// forward Euler integration
		p.velocity += dt * p.force / p.density;
		p.position += dt * p.velocity;

		// enforce boundary conditions
		if (p.position[0] - EPS < 0.f)
		{
			p.velocity[0] *= BOUND_DAMPING;
			p.position[0] = EPS;
		}
		if (p.position[0] + EPS > WINDOW_WIDTH)
		{
			p.velocity[0] *= BOUND_DAMPING;
			p.position[0] = WINDOW_WIDTH - EPS;
		}
		if (p.position[1] - EPS < 0.f)
		{
			p.velocity[1] *= BOUND_DAMPING;
			p.position[1] = EPS;
		}
		if (p.position[1] + EPS > WINDOW_HEIGHT)
		{
			p.velocity[1] *= BOUND_DAMPING;
			p.position[1] = WINDOW_HEIGHT - EPS;
		}
	}
}

void Solver::ComputeDensityPressure()
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		auto& pi = m_Particles[i];

		pi.density = 0.f;
		for (int j = 0; j < m_Particles.size(); ++j)
		{
			auto& pj = m_Particles[j];

			glm::vec2 rij = pj.position - pi.position;
			float r2 = glm::length2(rij);

			if (r2 < HSQ)
			{
				// this computation is symmetric
				pi.density += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.pressure = GAS_CONST * (pi.density - REST_DENS);
	}
}

void Solver::ComputeForces()
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		auto& pi = m_Particles[i];

		glm::vec2 fpress(0.f, 0.f);
		glm::vec2 fvisc(0.f, 0.f);
		for (int j = 0; j < m_Particles.size(); ++j)
		{
			auto& pj = m_Particles[j];

			if (&pi == &pj)
			{
				continue;
			}

			glm::vec2 rij = pj.position - pi.position;
			float r = glm::length(rij);

			if (r < H)
			{
				// compute pressure force contribution
				float mangitude = MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * SPIKY_GRAD * pow(H - r, 3);
				fpress += glm::normalize(-rij) * mangitude;

				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.density * VISC_LAP * (H - r);
			}
		}
		glm::vec2 fgrav = G * MASS / pi.density;
		pi.force = fpress + fvisc + fgrav;
	}
}

void Solver::SpatialParallelInitSPH()
{
	InitSPH();
}

void Solver::SpatialParallelComputeDensityPressure()
{
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		float pDensity = 0;

		Particle& pi = m_Particles[i];

		glm::ivec2 cell = SpatialHashTable::GetCell(pi, CELL_SIZE);

		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				glm::ivec2 adjacentCell = cell + glm::ivec2(x, y);
				if (adjacentCell.x < 0 || adjacentCell.y < 0) // boundary condition
					continue;

				uint32_t adjacentCellHash = SpatialHashTable::GetHash(adjacentCell);
				uint32_t thisCellsStartingIndex = m_ParticleHashTable[adjacentCellHash];
				if (thisCellsStartingIndex == SpatialHashTable::NO_PARTICLE)
					continue;

				while (thisCellsStartingIndex < m_Particles.size())
				{
					if (thisCellsStartingIndex == i)
					{
						thisCellsStartingIndex++;
						continue;
					}

					if (m_Particles[thisCellsStartingIndex].hash != adjacentCellHash)
						break;

					Particle& pj = m_Particles[thisCellsStartingIndex];

					float dist2 = glm::length2(pj.position - pi.position);
					if (dist2 >= HSQ) 
					{
						thisCellsStartingIndex++;
						continue;
					}

					pDensity += MASS * POLY6 * pow(HSQ - dist2, 3.f);

					thisCellsStartingIndex++;
				}
			}
		}

		// Include self density (as itself isn't included in neighbour)
		pi.density = pDensity + MASS * POLY6 * pow(HSQ, 3.f);

		// Calculate pressure
		float pPressure = GAS_CONST * (pi.density - REST_DENS);
		pi.pressure = pPressure;
	}
}

void Solver::SpatialParallelComputeForces()
{
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		Particle& pi = m_Particles[i];
		glm::ivec2 cell = SpatialHashTable::GetCell(pi, CELL_SIZE);

		glm::vec2 fpress(0.f, 0.f);
		glm::vec2 fvisc(0.f, 0.f);

		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				glm::ivec2 adjacentCell = cell + glm::ivec2(x, y);
				if (adjacentCell.x < 0 || adjacentCell.y < 0) // boundary condition
					continue;

				uint32_t adjacentCellHash = SpatialHashTable::GetHash(adjacentCell);
				uint32_t thisCellsStartingIndex = m_ParticleHashTable[adjacentCellHash];
				if (thisCellsStartingIndex == SpatialHashTable::NO_PARTICLE)
					continue;

				while (thisCellsStartingIndex < m_Particles.size())
				{
					if (thisCellsStartingIndex == i)
					{
						thisCellsStartingIndex++;
						continue;
					}

					if (m_Particles[thisCellsStartingIndex].hash != adjacentCellHash)
						break;

					Particle& pj = m_Particles[thisCellsStartingIndex];

					float dist2 = glm::length2(pj.position - pi.position);
					if (dist2 >= HSQ)
					{
						thisCellsStartingIndex++;
						continue;
					}

					//unit direction and length
					float dist = sqrt(dist2);
					glm::vec2 dir = glm::normalize(pj.position - pi.position);

					//apply pressure force
					float mangitude = MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * SPIKY_GRAD * pow(H - dist, 3);
					glm::vec2 pressureForce = mangitude * -dir;

					fpress += pressureForce;

					//apply viscosity force
					fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.density * VISC_LAP * (H - dist);

					thisCellsStartingIndex++;
				}
			}
		}

		glm::vec2 fgrav = G * MASS / pi.density;
		pi.force = fpress + fvisc + fgrav;
	}
}

void Solver::ParallelCalculateHashes()
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		m_Particles[i].hash = SpatialHashTable::GetHashFromParticle(m_Particles[i], CELL_SIZE);
	}
}

void Solver::SpatialParallelUpdate()
{
	Timer timer;

	// each particle gets assigned a hash corresponding to a particular cell in the grid
	ParallelCalculateHashes();

#ifdef PROFILE_TIMES
	std::cout << "Calculate Hash. " << timer.GetElapsed(true) << " ms\n";
#endif

	// sort particles by the particle's hash
	std::sort(
		m_Particles.begin(), m_Particles.end(),
		[&](const Particle& i, const Particle& j) {
			return i.hash < j.hash;
		}
	);

#ifdef PROFILE_TIMES
	std::cout << "Sort. " << timer.GetElapsed(true) << " ms\n";
#endif

	// m_ParticleHashTable[cell_hash] = starting particle index of that cell
	m_ParticleHashTable.resize(SpatialHashTable::TABLE_SIZE);
	SpatialHashTable::CreateNonAlloc(m_Particles, m_ParticleHashTable);

#ifdef PROFILE_TIMES
	std::cout << "Creating HashTable. " << timer.GetElapsed(true) << " ms\n";
#endif

#pragma region Debug

	// the following section debugs the spatial hash table
	//std::vector<uint32_t> availableHashes;
	//for (auto hash : m_ParticleHashTable)
	//{
	//	if (hash != SpatialHashTable::NO_PARTICLE)
	//		availableHashes.emplace_back(hash);
	//}

	//std::cout << "Total hashes " << availableHashes.size() << std::endl;

	// The following section tries to find a particle given a spawn ID (e.g. targetRawPid = 0)
	//constexpr int targetRawPid = 0;

	//int pid = -1;
	//for (int i = 0; i < m_Particles.size(); ++i)
	//{
	//	if (m_Particles[i].id == targetRawPid) {
	//		pid = i;
	//		break;
	//	}
	//}

	//assert(pid != -1);

	// color everything black
	//#pragma omp parallel for
	//for (int i = 0; i < m_Particles.size(); ++i)
	//{
	//	m_Particles[i].debugColor = 0;
	//}

	// UNCOMMENT BELOW TO DEBUG A PARTICULAR PARTICLE
	//std::vector<uint32_t> nearbyParticleIds = FindNearbyParticles(pid);
	//std::cout << nearbyParticleIds.size() << "\n";
	//#pragma omp parallel for
	//for (int i = 0; i < nearbyParticleIds.size(); ++i)
	//{
	//	m_Particles[nearbyParticleIds[i]].debugColor = 0.5f;
	//}
	//m_Particles[pid].debugColor = 1;

	// UNCOMMENT BELOW TO DEBUG SPATIAL BINS BY COLOR
	//#pragma omp parallel for
	//for (int i = 0; i < m_Particles.size(); ++i)
	//{
	//	m_Particles[i].debugColor = (float)SpatialHashTable::GetHashFromParticle(m_Particles[i], CELL_SIZE) / SpatialHashTable::TABLE_SIZE;
	//}
#pragma endregion

	SpatialParallelComputeDensityPressure();

#ifdef PROFILE_TIMES
	std::cout << "Compute Density Pressure. " << timer.GetElapsed(true) << " ms\n";
#endif

	SpatialParallelComputeForces();

#ifdef PROFILE_TIMES
	std::cout << "Compute Forces. " << timer.GetElapsed(true) << " ms\n";
#endif

	Integrate(DT);

#ifdef PROFILE_TIMES
	std::cout << "Integrate. " << timer.GetElapsed(true) << " ms\n";
#endif

#ifdef PROFILE_FRAME_TIME
	std::cout << "Solver total time. " << timer.GetElapsed(true) << " ms\n";
#endif
}

std::vector<uint32_t> Solver::FindNearbyParticles(int sortedPid)
{
	std::vector<uint32_t> particlesIndices;
	particlesIndices.reserve(25); // reserve an average amount m

	glm::ivec2 cell = SpatialHashTable::GetCell(m_Particles[sortedPid], CELL_SIZE);

	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			glm::ivec2 adjacentCell = cell + glm::ivec2(x, y);
			if (adjacentCell.x < 0 && adjacentCell.y < 0) // boundary condition
				continue;

			uint32_t adjacentCellHash = SpatialHashTable::GetHash(adjacentCell);
			uint32_t thisCellsStartingIndex = m_ParticleHashTable[adjacentCellHash];
			if (thisCellsStartingIndex == SpatialHashTable::NO_PARTICLE)
				continue;

			while (thisCellsStartingIndex < m_Particles.size())
			{
				if (thisCellsStartingIndex == sortedPid) 
				{
					thisCellsStartingIndex++;
					continue;
				}

				if (m_Particles[thisCellsStartingIndex].hash != adjacentCellHash)
					break;

				particlesIndices.emplace_back(thisCellsStartingIndex);

				thisCellsStartingIndex++;
			}
		}
	}

	return particlesIndices;
}

void Solver::KernelInitSPH()
{
}

void Solver::KernelComputeDensityPressure()
{
}

void Solver::KernelComputeForces()
{
}

void Solver::KernelIntegrate(float dt)
{
}
