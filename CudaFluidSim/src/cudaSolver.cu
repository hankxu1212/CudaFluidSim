#include "cudaSolver.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <device_launch_parameters.h>

// host handles
d_Particle* d_Particles;

// device constants
//__constant__ float2 cuG;
//__constant__ uint32_t cuWINDOW_WIDTH;
//__constant__ uint32_t cuWINDOW_HEIGHT;

__global__ void KernelComputeDensityPressure(d_Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& pi = particles[i].position_density_pressure;
		float density = 0.f;
		for (int j = 0; j < NUM_PARTICLES; ++j)
		{
			auto& pj = particles[j].position_density_pressure;
			float2 rij = make_float2(pj.x - pi.x, pj.y - pi.y);
			float r2 = rij.x*rij.x + rij.y*rij.y;

			if (r2 < HSQ)
			{
				// this computation is symmetric
				density += MASS * POLY6 * pow(HSQ - r2, 3.f);
			}
		}
		pi.z = density;
		pi.w = GAS_CONST * (density - REST_DENS);
	}
}

__global__ void KernelComputeForces(d_Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& pi = particles[i];

		float2 fpress = make_float2(0.f, 0.f);
		float2 fvisc = make_float2(0.f, 0.f);
		for (int j = 0; j < N; ++j)
		{
			auto& pj = particles[j];

			if (&pi == &pj) { continue; }

			float2 rij = make_float2(pj.position_density_pressure.x - pi.position_density_pressure.x, pj.position_density_pressure.y - pi.position_density_pressure.y);
			float r = sqrt(rij.x * rij.x + rij.y * rij.y);

			if (r < H)
			{
				// compute pressure force contribution
				float magnitude = MASS * (pi.position_density_pressure.w + pj.position_density_pressure.w) / (2.0f * pj.position_density_pressure.z) * SPIKY_GRAD * pow(H - r, 3.f);
				fpress = make_float2(fpress.x - rij.x/r * magnitude, fpress.y - rij.y/r * magnitude);

				// compute viscosity force contribution
				fvisc = make_float2(
					fvisc.x + VISC * MASS * (pj.velocity_force.x - pi.velocity_force.x) / pj.position_density_pressure.z * VISC_LAP * (H - r),
					fvisc.y + VISC * MASS * (pj.velocity_force.y - pi.velocity_force.y) / pj.position_density_pressure.z * VISC_LAP * (H - r)
				);
			}
		}
		float2 fgrav = make_float2(cuGx * MASS / pi.position_density_pressure.z, cuGy * MASS / pi.position_density_pressure.z);
		pi.velocity_force.z = fpress.x + fvisc.x + fgrav.x;
		pi.velocity_force.w = fpress.y + fvisc.y + fgrav.y;
	}
}

__global__ void KernelIntegrate(float dt, d_Particle* particles, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		auto& p = particles[i];

		// forward Euler integration
		p.velocity_force.x += dt * p.velocity_force.z / p.position_density_pressure.z;
		p.velocity_force.y += dt * p.velocity_force.w / p.position_density_pressure.z;
		p.position_density_pressure.x += dt * p.velocity_force.x;
		p.position_density_pressure.y += dt * p.velocity_force.y;

		// enforce boundary conditions
		if (p.position_density_pressure.x - EPS < 0.f)
		{
			p.velocity_force.x *= BOUND_DAMPING;
			p.position_density_pressure.x = EPS;
		}
		if (p.position_density_pressure.x + EPS > cuWINDOW_WIDTH)
		{
			p.velocity_force.x *= BOUND_DAMPING;
			p.position_density_pressure.x = cuWINDOW_WIDTH - EPS;
		}
		if (p.position_density_pressure.y - EPS < 0.f)
		{
			p.velocity_force.y *= BOUND_DAMPING;
			p.position_density_pressure.y = EPS;
		}
		if (p.position_density_pressure.y + EPS > cuWINDOW_HEIGHT)
		{
			p.velocity_force.y *= BOUND_DAMPING;
			p.position_density_pressure.y = cuWINDOW_HEIGHT - EPS;
		}
	}
}

void DeviceInitSPH(d_Particle* hostParticles)
{
	cudaMalloc(&d_Particles, NUM_PARTICLES * sizeof(d_Particle));
	cudaMemcpy(d_Particles, hostParticles, NUM_PARTICLES * sizeof(d_Particle), cudaMemcpyHostToDevice);

	//cudaMemcpyToSymbol(&cuG, &G, sizeof(glm::vec2));
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

void DeviceSync(d_Particle* hostParticles, size_t count)
{
	cudaMemcpy(hostParticles, d_Particles, count * sizeof(d_Particle), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}
