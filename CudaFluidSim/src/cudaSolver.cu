#include "cudaSolver.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

// host handles
Particle* d_Particles;

// device constants
__constant__ glm::vec2 cuG;
__constant__ unsigned int cuWINDOW_WIDTH;
__constant__ unsigned int cuWINDOW_HEIGHT;

__global__ void KernelComputeDensityPressure(Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& pi = particles[i];
		float density = 0.f;
		for (int j = 0; j < NUM_PARTICLES; ++j)
		{
			auto& pj = particles[j];
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
				float mangitude = MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * SPIKY_GRAD * pow(H - r, 3);
				fpress += glm::normalize(-rij) * mangitude;

				// compute viscosity force contribution
				fvisc += VISC * MASS * (pj.velocity - pi.velocity) / pj.density * VISC_LAP * (H - r);
			}
		}
		glm::vec2 fgrav = cuG * MASS / pi.density;
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
		if (p.position[0] - EPS < 0.f)
		{
			p.velocity[0] *= BOUND_DAMPING;
			p.position[0] = EPS;
		}
		if (p.position[0] + EPS > cuWINDOW_WIDTH)
		{
			p.velocity[0] *= BOUND_DAMPING;
			p.position[0] = cuWINDOW_WIDTH - EPS;
		}
		if (p.position[1] - EPS < 0.f)
		{
			p.velocity[1] *= BOUND_DAMPING;
			p.position[1] = EPS;
		}
		if (p.position[1] + EPS > cuWINDOW_HEIGHT)
		{
			p.velocity[1] *= BOUND_DAMPING;
			p.position[1] = cuWINDOW_HEIGHT - EPS;
		}
	}
}

void DeviceInitSPH(Particle* hostParticles, uint32_t windowHeight, uint32_t windowWidth)
{
	cudaMalloc(&d_Particles, NUM_PARTICLES * sizeof(Particle));
	cudaMemcpy(d_Particles, hostParticles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(&cuG, &G, sizeof(glm::vec2));
	cudaMemcpyToSymbol(&cuWINDOW_HEIGHT, &windowHeight, sizeof(unsigned int));
	cudaMemcpyToSymbol(&cuWINDOW_WIDTH, &windowWidth, sizeof(unsigned int));
}

void DispatchComputeDensityPressure()
{
	dim3 blockDim(256, 1);
	dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
	KernelComputeDensityPressure <<<gridDim, blockDim>>> (d_Particles, NUM_PARTICLES);
	// probably doesn't properly compile and stuff, but am focusing on writing the logic before figuring out how
	// the syntax works
}

void DispatchComputeForces()
{
	dim3 blockDim(256, 1);
	dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
	KernelComputeForces <<<gridDim, blockDim>>> (d_Particles, NUM_PARTICLES);
}

void DispatchIntegrate(float dt)
{
	dim3 blockDim(256, 1);
	dim3 gridDim((NUM_PARTICLES + blockDim.x - 1) / blockDim.x);
	KernelIntegrate <<<gridDim, blockDim>>> (dt, d_Particles, NUM_PARTICLES);
}

void DeviceCleanup()
{
	if (d_Particles != nullptr)
		cudaFree(d_Particles);
}

void DeviceSync(Particle* hostParticles, size_t count)
{
	cudaMemcpy(hostParticles, d_Particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}
