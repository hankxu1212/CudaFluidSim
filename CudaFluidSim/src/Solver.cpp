#include "Solver.hpp"
#include "math/Math.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

#include <omp.h>

#include "utils/Time.hpp"

// "Particle-Based Fluid Simulation for Interactive Applications" by Müller et al.
// solver parameters
const static glm::vec2 G(0.f, 10.f);   // external (gravitational) forces
const static float REST_DENS = 300.f;  // rest density
const static float GAS_CONST = 2000.f; // const for equation of state
const static float H = 8.f;		   // kernel radius
const static float HSQ = H * H;		   // radius^2 for optimization
const static float MASS = 2.5f;		   // assume all particles have the same mass
const static float VISC = 200.f;	   // viscosity constant
const static float DT = 0.001f;

// smoothing kernels defined in Müller and their gradients
// adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
const static float POLY6 = 4.f / (glm::pi<float>() * pow(H, 8.f));
const static float SPIKY_GRAD = -10.f / (glm::pi<float>() * pow(H, 5.f));
const static float VISC_LAP = 40.f / (glm::pi<float>() * pow(H, 5.f));

// interaction
const static int NUM_PARTICLES = 1000;

// rendering projection parameters
#define VIEW_HEIGHT Window::Get()->m_Data.Height
#define VIEW_WIDTH Window::Get()->m_Data.Width

// simulation parameters
const static float EPS = H; // boundary epsilon
const static float BOUND_DAMPING = -0.5f;

Solver::Solver()
{
	omp_set_num_threads(8);

	InitSPH();
}

Solver::~Solver()
{
}

void Solver::Update()
{
	ComputeDensityPressure();
	ComputeForces();
	Integrate(DT);
}

void Solver::InitSPH()
{
	for (float y = EPS; y < VIEW_HEIGHT - EPS * 2.f; y += H)
	{
		for (float x = VIEW_WIDTH / 4; x <= VIEW_WIDTH / 2; x += H)
		{
			if (m_Particles.size() < NUM_PARTICLES)
			{
				float jitter = static_cast<float>(Math::Random()) / static_cast<float>(RAND_MAX);
				m_Particles.push_back(Particle(x + jitter, y));
			}
			else
			{
				return;
			}
		}
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
		p.v += dt * p.f / p.rho;
		p.x += dt * p.v;

		// enforce boundary conditions
		if (p.x[0] - EPS < 0.f)
		{
			p.v[0] *= BOUND_DAMPING;
			p.x[0] = EPS;
		}
		if (p.x[0] + EPS > VIEW_WIDTH)
		{
			p.v[0] *= BOUND_DAMPING;
			p.x[0] = VIEW_WIDTH - EPS;
		}
		if (p.x[1] - EPS < 0.f)
		{
			p.v[1] *= BOUND_DAMPING;
			p.x[1] = EPS;
		}
		if (p.x[1] + EPS > VIEW_HEIGHT)
		{
			p.v[1] *= BOUND_DAMPING;
			p.x[1] = VIEW_HEIGHT - EPS;
		}
	}
}

void Solver::ComputeDensityPressure()
{
	#pragma omp parallel for
	for (int i = 0; i < m_Particles.size(); ++i)
	{
		auto& pi = m_Particles[i];

		pi.rho = 0.f;
		for (int j = 0; j < m_Particles.size(); ++j)
		{
			auto& pj = m_Particles[j];

			glm::vec2 rij = pj.x - pi.x;
			float r2 = glm::length2(rij);

			if (r2 < HSQ)
			{
				// this computation is symmetric
				pi.rho += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.p = GAS_CONST * (pi.rho - REST_DENS);
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

			glm::vec2 rij = pj.x - pi.x;
			float r = glm::length(rij);

			if (r < H)
			{
				// compute pressure force contribution
				fpress += glm::normalize(-rij) * MASS * (pi.p + pj.p) / (2.f * pj.rho) * SPIKY_GRAD * pow(H - r, 3.f);
				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.v - pi.v) / pj.rho * VISC_LAP * (H - r);
			}
		}
		glm::vec2 fgrav = G * MASS / pi.rho;
		pi.f = fpress + fvisc + fgrav;
	}
}
