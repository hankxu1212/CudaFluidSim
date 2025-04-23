#pragma once

#ifndef __CUDA_SOLVER_H__
#define __CUDA_SOLVER_H__

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"
#endif

#include "Particle.hpp"

void DeviceInitSPH(Particle* hostParticles, uint32_t windowHeight, uint32_t windowWidth);
void DispatchComputeDensityPressure();
void DispatchComputeForces();
void DispatchIntegrate(float dt);
void DeviceCleanup();

void DeviceSync(Particle* hostParticles, size_t count);

#endif