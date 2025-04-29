#pragma once

constexpr float PI = 3.14159265358979323846f;
constexpr int H = 4;				// kernel radius
constexpr int CELL_SIZE = H * 2;		// spatial grid size
constexpr float HSQ = H * H;		   // radius^2 for optimization
constexpr float REST_DENS = 300.f;  // rest density
constexpr float GAS_CONST = 2000.f; // const for equation of state
constexpr float MASS = 2.5f;		   // assume all particles have the same mass
constexpr float VISC = 1500.f;	   // viscosity constant
constexpr float DT = 0.0007f;       // simulation delta time

// workaround since idk why the values aren't copying over
constexpr uint32_t cuWINDOW_HEIGHT = 1000;
constexpr uint32_t cuWINDOW_WIDTH = 1000;
constexpr float cuGx = 0.f;
constexpr float cuGy = 10.f;

const static glm::vec2 G(0.f, 10.f);   // external (gravitational) forces

constexpr int NUM_PARTICLES = 40000;

constexpr float EPS = H; // boundary epsilon
constexpr float BOUND_DAMPING = -0.8f;


template <typename T>
constexpr T power_constexpr(T base, unsigned exponent) noexcept {
    // Compile-time optimized path
    T result = 1;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result *= base;
        }
        base *= base;
        exponent /= 2;
    }
    return result;
}

constexpr float compute_poly6_constant(float h) noexcept {
    return 4.0f / (PI * power_constexpr(h, 8));
}

constexpr float compute_spiky_grad_constant(float h) noexcept {
    return -10.0f / (PI * power_constexpr(h, 5));
}

constexpr float compute_visc_lap_constant(float h) noexcept {
    return 40.0f / (PI * power_constexpr(h, 5));
}

// Usage (compile-time evaluation):
static constexpr float POLY6 = compute_poly6_constant(H);
static constexpr float SPIKY_GRAD = compute_spiky_grad_constant(H);
static constexpr float VISC_LAP = compute_visc_lap_constant(H);