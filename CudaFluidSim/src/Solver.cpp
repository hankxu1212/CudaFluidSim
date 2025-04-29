#include "Solver.hpp"

#include "cudaSolver.cuh"

#include "math/Math.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

#include <omp.h>

#include "utils/Time.hpp"
#include "utils/Timer.hpp"

#include <execution>

#include <imgui.h>

#define WINDOW_HEIGHT Window::Get()->m_Data.Height
#define WINDOW_WIDTH Window::Get()->m_Data.Width

//#define PROFILE_TIMES

Solver* Solver::s_Instance = nullptr;

Solver::Solver()
{
	s_Instance = this;

	OnRestart();
}

Solver::~Solver()
{
	DeviceCleanup();
}

void Solver::OnUpdate()
{
	if (Paused)
		return;

	if (Restart) {
		OnRestart();
		Restart = false;
	}

	Timer totalTimer;

	auto accelerationMode = Application::GetSpecification().accelerationMode;

	switch (accelerationMode)
	{
	case ApplicationSpecification::Naive:
		ComputeDensityPressure();
		//std::cout << "After ComputeDensityPressure " << m_Particles[0].density << " " << m_Particles[0].pressure << std::endl;
		ComputeForces();
		//std::cout << "After ComputeForces " << m_Particles[0].force[0] << " " << m_Particles[0].force[1] << std::endl;
		Integrate(DT);
		//std::cout << "After Integrate " << m_Particles[0].position[0] << " " << m_Particles[0].position[1] << std::endl;

		LastFrameUpdateTime = totalTimer.GetElapsed(true);
		break;

	case ApplicationSpecification::Spatial:
	case ApplicationSpecification::SpatialCombined:
		SpatialParallelUpdate();

		LastFrameUpdateTime = totalTimer.GetElapsed(true);
		break;

	case ApplicationSpecification::GPU:
		DispatchComputeDensityPressure();
		/*DeviceSync(m_dParticles.data(), NUM_PARTICLES);
		std::cout << "After ComputeDensityPressure " << m_dParticles[0].position_density_pressure.z << " " << m_dParticles[0].position_density_pressure.w << std::endl;*/
		DispatchComputeForces();
		/*DeviceSync(m_dParticles.data(), NUM_PARTICLES);
		std::cout << "After ComputeForces " << m_dParticles[0].velocity_force.z << " " << m_dParticles[0].velocity_force.w << std::endl;*/
		DispatchIntegrate(DT);
		DeviceSync(m_dParticles.data(), NUM_PARTICLES);
		//std::cout << "After Integrate " << m_dParticles[0].position_density_pressure.x << " " << m_dParticles[0].position_density_pressure.y << std::endl;

		LastFrameUpdateTime = totalTimer.GetElapsed(true);

		for (int i = 0; i < NUM_PARTICLES; ++i)
		{
			auto& mp = m_Particles[i];
			auto& dp = m_dParticles[i];
			mp.position[0] = dp.position_density_pressure.x;
			mp.position[1] = dp.position_density_pressure.y;
			mp.density = dp.position_density_pressure.z;
			mp.pressure = dp.position_density_pressure.w;
			mp.velocity[0] = dp.velocity_force.x;
			mp.velocity[1] = dp.velocity_force.y;
			mp.force[0] = dp.velocity_force.z;
			mp.force[1] = dp.velocity_force.w;
			//m_Particles[i] = Particle(m_dParticles[i]);
		}

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
		Paused = !Paused;
		break;
	}

	return false;
}


void Solver::OnRestart()
{
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
	case ApplicationSpecification::SpatialCombined:
		std::cout << "Acceleration method: Spatial Hashing\n";
		SpatialParallelInitSPH();
		break;

	case ApplicationSpecification::GPU:
		std::cout << "Acceleration method: GPU\n";
		DeviceCleanup();
		InitSPH();
		DeviceInitSPH(m_dParticles.data());
		break;
	}
}

// Inside your Solver class implementation
void Solver::OnImGuiRender()
{
	if (Restart)
		return;

	ImGui::BulletText("Number of Particles: %d", NUM_PARTICLES);
	ImGui::BulletText("Smoothing Kernel: %d", H);

	if (Paused)
		ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Solver: Paused");
	else
		ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Solver: Running");

	if (ImGui::Button(Paused ? "Resume" : "Pause"))
		Paused = !Paused;

	if (ImGui::Button("Restart Simulation"))
		Restart = true; // consumed in next update loop
}


void Solver::InitSPH()
{
	for (int i = 0; i < NUM_PARTICLES; ++i) 
	{
		float angle = Math::Random(0, 2.0f * 3.1415f);
		float r = Math::Random();

		float x = WINDOW_HEIGHT / 3 + WINDOW_HEIGHT / 3 * r * cos(angle) + WINDOW_HEIGHT / 5;
		float y = WINDOW_HEIGHT / 3 + WINDOW_HEIGHT / 3 * r * sin(angle);

		m_dParticles[i] = d_Particle(x, y);
		m_Particles[i] = Particle(x, y);
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

void Solver::LeapfrogKick(float dt)
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		Particle& p = m_Particles[i];
		// Update half-step velocity with the new force.
		p.velocityHalf += dt * (p.force / p.density);

		// Optionally update full velocity for diagnostics or output:
		// (This recovers the full-step velocity at t + dt.)
		p.velocity = p.velocityHalf - 0.5f * dt * (p.force / p.density);
	}
}

void Solver::LeapfrogDrift(float dt)
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		Particle& p = m_Particles[i];
		p.position += dt * p.velocityHalf;

		// Enforce boundary conditions after drift:
		if (p.position.x - EPS < 0.f)
		{
			p.velocityHalf.x *= BOUND_DAMPING;
			p.position.x = EPS;
		}
		if (p.position.x + EPS > WINDOW_WIDTH)
		{
			p.velocityHalf.x *= BOUND_DAMPING;
			p.position.x = WINDOW_WIDTH - EPS;
		}
		if (p.position.y - EPS < 0.f)
		{
			p.velocityHalf.y *= BOUND_DAMPING;
			p.position.y = EPS;
		}
		if (p.position.y + EPS > WINDOW_HEIGHT)
		{
			p.velocityHalf.y *= BOUND_DAMPING;
			p.position.y = WINDOW_HEIGHT - EPS;
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

		#pragma omp parallel for reduction(+: pDensity)
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

		#pragma omp parallel for
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

void Solver::SpatialParallelComputeCombined()
{
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic, 32) 
		for (int i = 0; i < NUM_PARTICLES; ++i)
		{
			float pDensity = 0;
			Particle& pi = m_Particles[i];
			const glm::ivec2 cell = SpatialHashTable::GetCell(pi, CELL_SIZE);

			#pragma omp parallel for reduction(+:pDensity) schedule(static)
			for (int xy = 0; xy < 9; xy++) {
				const int x = xy / 3 - 1;  // Converts 0-8 to -1 to +1
				const int y = xy % 3 - 1;
				const glm::ivec2 adjacentCell = cell + glm::ivec2(x, y);
				if (adjacentCell.x < 0 || adjacentCell.y < 0) continue;

				const uint32_t adjacentCellHash = SpatialHashTable::GetHash(adjacentCell);
				uint32_t idx = m_ParticleHashTable[adjacentCellHash];
				if (idx == SpatialHashTable::NO_PARTICLE) continue;

				while (idx < NUM_PARTICLES) 
				{
					if (m_Particles[idx].hash != adjacentCellHash) break;
					if (idx != i) 
					{
						const float dist2 = glm::length2(m_Particles[idx].position - pi.position);
						if (dist2 < HSQ) {
							pDensity += MASS * POLY6 * pow(HSQ - dist2, 3.f);
						}
					}
					idx++;
				}
			}

			// Include self density (as itself isn't included in neighbour)
			pi.density = pDensity + MASS * POLY6 * pow(HSQ, 3.f);
			pi.pressure = GAS_CONST * (pi.density - REST_DENS);;
		}

		#pragma omp for schedule(dynamic, 32) nowait
		for (int i = 0; i < NUM_PARTICLES; ++i)
		{
			Particle& pi = m_Particles[i];
			const glm::ivec2 cell = SpatialHashTable::GetCell(pi, CELL_SIZE);

			float fpress_x = 0.f, fpress_y = 0.f;
			float fvisc_x = 0.f, fvisc_y = 0.f;

			#pragma omp parallel for reduction(+:fpress_x,fpress_y,fvisc_x,fvisc_y) schedule(static)
			for (int xy = 0; xy < 9; xy++) {
				const int x = xy / 3 - 1;  // Converts 0-8 to -1 to +1
				const int y = xy % 3 - 1;
				const glm::ivec2 adjacentCell = cell + glm::ivec2(x, y);
				if (adjacentCell.x < 0 || adjacentCell.y < 0) continue;

				const uint32_t adjacentCellHash = SpatialHashTable::GetHash(adjacentCell);
				uint32_t idx = m_ParticleHashTable[adjacentCellHash];
				if (idx == SpatialHashTable::NO_PARTICLE) continue;

				while (idx < NUM_PARTICLES) {
					if (m_Particles[idx].hash != adjacentCellHash) break;
					if (idx != i) {
						const Particle& pj = m_Particles[idx];
						float dist2 = glm::length2(pj.position - pi.position);
						if (dist2 < HSQ) {
							float dist = sqrt(dist2);
							glm::vec2 dir = (dist > 0) ? (pj.position - pi.position) / dist : glm::vec2(0.f);

							// Pressure force
							float magnitude = -MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * SPIKY_GRAD * pow(H - dist, 3);

							auto fpress = magnitude * dir;
							fpress_x += fpress.x;
							fpress_y += fpress.y;

							auto fvisc = VISC * MASS * (pj.velocity - pi.velocity) / pj.density * VISC_LAP * (H - dist);
							fvisc_x += fvisc.x;
							fvisc_y += fvisc.y;
						}
					}
					idx++;
				}
			}

			glm::vec2 fgrav = G * MASS / pi.density;
			pi.force = glm::vec2(fpress_x, fpress_y) + glm::vec2(fvisc_x, fvisc_y) + fgrav;
		}
	}
}

static const int SERIAL_THRESHOLD = 1024;

template<typename T, typename Compare>
void quicksort_omp(T* data, int left, int right, Compare comp) {
	if (left >= right) return;

	int size = right - left + 1;
	if (size < SERIAL_THRESHOLD) {
		std::sort(data + left, data + right + 1, comp);
		return;
	}

	// pick pivot
	T pivot = data[left + size / 2];
	int i = left, j = right;
	while (i <= j) {
		while (comp(data[i], pivot)) ++i;
		while (comp(pivot, data[j])) --j;
		if (i <= j) {
			std::swap(data[i], data[j]);
			++i; --j;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		quicksort_omp(data, left, j, comp);
#pragma omp section
		quicksort_omp(data, i, right, comp);
	}
}

void Solver::SpatialParallelUpdate()
{
	switch (Application::GetSpecification().integrationMode)
	{
	case ApplicationSpecification::Euler:
		break;
	case ApplicationSpecification::LeapFrog:
		LeapfrogDrift(DT);
		break;
	}

#ifdef PROFILE_TIMES
	Timer timer;
#endif

	// each particle gets assigned a hash corresponding to a particular cell in the grid
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		m_Particles[i].hash = SpatialHashTable::GetHashFromParticle(m_Particles[i], CELL_SIZE);
	}

#ifdef PROFILE_TIMES
	std::cout << "Calculate Hash. " << timer.GetElapsed(true) << " ms\n";
#endif

	// sort particles in parallel by the particle's hash
	auto cmp = [&](const Particle& i, const Particle& j) { return i.hash < j.hash; };

	//std::sort(m_Particles.begin(), m_Particles.end(), cmp);
	#pragma omp parallel
	#pragma omp single nowait
	{
		quicksort_omp(m_Particles.data(), 0, int(m_Particles.size()) - 1, cmp);
	}

#ifdef PROFILE_TIMES
	std::cout << "Sort. " << timer.GetElapsed(true) << " ms\n";
#endif

	// m_ParticleHashTable[cell_hash] = starting particle index of that cell
	SpatialHashTable::CreateNonAlloc(m_Particles.data(), NUM_PARTICLES, m_ParticleHashTable.data());

#ifdef PROFILE_TIMES
	std::cout << "Creating HashTable. " << timer.GetElapsed(true) << " ms\n";
#endif

	switch (Application::GetSpecification().accelerationMode)
	{
	case ApplicationSpecification::Spatial:
		SpatialParallelComputeDensityPressure();
		SpatialParallelComputeForces();
		break;

	case ApplicationSpecification::SpatialCombined:
		SpatialParallelComputeCombined();
		break;
	}

#ifdef PROFILE_TIMES
	std::cout << "Compute Density Pressure and Forces. " << timer.GetElapsed(true) << " ms\n";
#endif

	switch (Application::GetSpecification().integrationMode)
	{
	case ApplicationSpecification::Euler:
		Integrate(DT);
		break;
	case ApplicationSpecification::LeapFrog:
		LeapfrogKick(DT);
		break;
	}

#ifdef PROFILE_TIMES
	std::cout << "Integrate. " << timer.GetElapsed(true) << " ms\n";
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