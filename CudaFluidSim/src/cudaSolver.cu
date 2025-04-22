#include "Solver.hpp"
#include "math/Math.hpp"
#include "cudaSolver.cuh"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

#include "Particle.hpp"

#define WINDOW_HEIGHT Window::Get()->m_Data.Height
#define WINDOW_WIDTH Window::Get()->m_Data.Width
#define m_Particles Solver::Get()->m_Particles

__constant__ glm::vec2 cuG;
__constant__ int cuH;
__constant__ int cuCELL_SIZE;
__constant__ float cuHSQ;
__constant__ float cuREST_DENS;
__constant__ float cuGAS_CONST;
__constant__ float cuMASS;
__constant__ float cuVISC;
__constant__ float cuDT;
__constant__ float cuPOLY6;
__constant__ float cuSPIKY_GRAD;
__constant__ float cuVISC_LAP;
__constant__ float cuEPS;
__constant__ float cuBOUND_DAMPING;
__constant__ unsigned int cuWINDOW_WIDTH;
__constant__ unsigned int cuWINDOW_HEIGHT;


__global__ void KernelComputeDensityPressure(Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& pi = particles[i];
		float density = 0.f;
		for (int j = 0; j < N; ++j)
		{
			auto& pj = particles[j];
			glm::vec2 rij = pj.position - pi.position;
			float r2 = glm::length2(rij);

			if (r2 < cuHSQ)
			{
				// this computation is symmetric
				pi.density += cuMASS * cuPOLY6 * pow(cuHSQ - r2, 3.f);
			}
		}
		pi.pressure = cuGAS_CONST * (pi.density - cuREST_DENS);
	}
}

__global__ void KernelComputeForces(Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& pi = particles[i];

		glm::vec2 fpress(0.f, 0.f);
		glm::vec2 fvisc(0.f, 0.f);
		for (int j = 0; j < N; ++j)
		{
			auto& pj = particles[j];

			if (&pi == &pj)
			{
				continue;
			}

			glm::vec2 rij = pj.position - pi.position;
			float r = glm::length(rij);

			if (r < H)
			{
				// compute pressure force contribution
				float mangitude = cuMASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * cuSPIKY_GRAD * pow(H - r, 3);
				fpress += glm::normalize(-rij) * mangitude;

				// compute viscosity force contribution
				fvisc += cuVISC * cuMASS * (pj.velocity - pi.velocity) / pj.density * cuVISC_LAP * (H - r);
			}
		}
		glm::vec2 fgrav = cuG * cuMASS / pi.density;
		pi.force = fpress + fvisc + fgrav;
	}
}

__global__ void KernelIntegrate(float dt, Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& p = particles[i];

		// forward Euler integration
		p.velocity += dt * p.force / p.density;
		p.position += dt * p.velocity;

		// enforce boundary conditions
		if (p.position[0] - cuEPS < 0.f)
		{
			p.velocity[0] *= cuBOUND_DAMPING;
			p.position[0] = cuEPS;
		}
		if (p.position[0] + cuEPS > cuWINDOW_WIDTH)
		{
			p.velocity[0] *= cuBOUND_DAMPING;
			p.position[0] = cuWINDOW_WIDTH - cuEPS;
		}
		if (p.position[1] - cuEPS < 0.f)
		{
			p.velocity[1] *= cuBOUND_DAMPING;
			p.position[1] = cuEPS;
		}
		if (p.position[1] + cuEPS > cuWINDOW_HEIGHT)
		{
			p.velocity[1] *= cuBOUND_DAMPING;
			p.position[1] = cuWINDOW_HEIGHT - cuEPS;
		}
	}
}

void CUDAInitSPH()
{
	for (int i = 0; i < NUM_PARTICLES; ++i)
	{
		float angle = Math::Random(0, 2.0f * 3.1415f);
		float r = Math::Random();

		float x = WINDOW_HEIGHT / 3 + WINDOW_HEIGHT / 3 * r * cos(angle) + WINDOW_HEIGHT / 5;
		float y = WINDOW_HEIGHT / 3 + WINDOW_HEIGHT / 3 * r * sin(angle);

		m_Particles[i] = Particle(x, y);
	}

	std::cout << "Initializing " << m_Particles.size() << " particles" << std::endl;

	cudaMalloc(&d_Particles, m_Particles.size() * sizeof(Particle));
	cudaMemcpy(d_Particles, m_Particles.data(), m_Particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(&cuG, &G, sizeof(glm::vec2));
	cudaMemcpyToSymbol(&cuH, &H, sizeof(int));
	cudaMemcpyToSymbol(&cuCELL_SIZE, &CELL_SIZE, sizeof(int));
	cudaMemcpyToSymbol(&cuHSQ, &HSQ, sizeof(float));
	cudaMemcpyToSymbol(&cuREST_DENS, &REST_DENS, sizeof(float));
	cudaMemcpyToSymbol(&cuGAS_CONST, &GAS_CONST, sizeof(float));
	cudaMemcpyToSymbol(&cuMASS, &MASS, sizeof(float));
	cudaMemcpyToSymbol(&cuVISC, &VISC, sizeof(float));
	cudaMemcpyToSymbol(&cuDT, &DT, sizeof(float));
	cudaMemcpyToSymbol(&cuPOLY6, &POLY6, sizeof(float));
	cudaMemcpyToSymbol(&cuSPIKY_GRAD, &SPIKY_GRAD, sizeof(float));
	cudaMemcpyToSymbol(&cuVISC_LAP, &VISC_LAP, sizeof(float));
	cudaMemcpyToSymbol(&cuEPS, &EPS, sizeof(float));
	cudaMemcpyToSymbol(&cuBOUND_DAMPING, &BOUND_DAMPING, sizeof(float));
	cudaMemcpyToSymbol(&cuWINDOW_HEIGHT, &WINDOW_HEIGHT, sizeof(unsigned int));
	cudaMemcpyToSymbol(&cuWINDOW_WIDTH, &WINDOW_WIDTH, sizeof(unsigned int));
}

void DispatchComputeDensityPressure()
{
	dim3 blockDim(256, 1);
	dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
	KernelComputeDensityPressure <<<gridDim, blockDim>>> (d_Particles, m_Particles.size());
	// probably doesn't properly compile and stuff, but am focusing on writing the logic before figuring out how
	// the syntax works
}

void DispatchComputeForces()
{
	dim3 blockDim(256, 1);
	dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
	KernelComputeForces <<<gridDim, blockDim>>> (d_Particles, m_Particles.size());
}

void DispatchIntegrate(float dt)
{
	dim3 blockDim(256, 1);
	dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
	KernelIntegrate <<<gridDim, blockDim>>> (dt, d_Particles, m_Particles.size());
}