// Includes, CUDA error checking, physical constants and boundary tags
// Part of cylinder_fem_amr; included in this order by cylinder_fem_amr.cu.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>   // __gnu_parallel::sort (OpenMP multiway mergesort)
#endif

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error at %s:%d in %s: %s\n", __FILE__,         \
                    __LINE__, __func__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// ============================================================================
// Physical and Simulation Constants (DFG 2D-2 Cylinder Benchmark)
// ============================================================================
namespace SimConfig {
    constexpr double L_CHANNEL  = 2.20;   // Channel length [m]
    constexpr double H_CHANNEL  = 0.41;   // Channel height [m]
    constexpr double X_CYL      = 0.20;   // Cylinder center X [m]
    constexpr double Y_CYL      = 0.20;   // Cylinder center Y [m]
    constexpr double R_CYL      = 0.05;   // Cylinder radius [m] (D = 0.10 m)
    constexpr double U_MAX      = 1.50;   // Peak inlet velocity [m/s] (U_bar = 1.0 m/s)
    constexpr double U_MEAN     = 1.00;   // Mean velocity (2/3 * U_MAX) [m/s]
    constexpr double NU         = 1.0e-3; // Kinematic viscosity [m^2/s] (Re = 100)
    constexpr double RHO        = 1.00;   // Fluid density [kg/m^3]
    constexpr double DT_DEFAULT = 1.0e-3; // Time step size [s]
    constexpr double PI         = 3.14159265358979323846;
}

// Peak inflow velocity at time t: constant (DFG 2D-2) or U_MAX sin(pi t / 8) (DFG 2D-3)
__host__ __device__ inline double inflow_umax(double t, bool ramp_2d3) {
    return ramp_2d3 ? SimConfig::U_MAX * sin(SimConfig::PI * t / 8.0) : SimConfig::U_MAX;
}

// Boundary Condition Identifiers
enum BoundaryTag : int {
    BC_INTERIOR = 0,
    BC_INLET    = 1,
    BC_OUTLET   = 2,
    BC_WALL     = 3,
    BC_CYLINDER = 4
};
