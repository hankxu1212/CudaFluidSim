#pragma once
#ifndef __CUDA_SOLVER_H__
#define __CUDA_SOLVER_H__

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/norm.hpp"
#endif

//glm::vec2 cuG;
//int cuH;
//int cuCELL_SIZE;
//float cuHSQ;
//float cuREST_DENS;
//float cuGAS_CONST;
//float cuMASS;
//float cuVISC;
//float cuDT;
//float cuPOLY6;
//float cuSPIKY_GRAD;
//float cuVISC_LAP;
//float cuEPS;
//float cuBOUND_DAMPING;
//unsigned int cuWINDOW_WIDTH;
//unsigned int cuWINDOW_HEIGHT;

Particle* d_Particles;

void CUDAInitSPH();
void DispatchComputeDensityPressure();
void DispatchComputeForces();
void DispatchIntegrate(float dt);

#endif