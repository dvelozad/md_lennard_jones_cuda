#ifndef PARTICLE_H
#define PARTICLE_H

#include "system.h"
#include <curand_kernel.h>

__device__ double Lx_d, Ly_d, Lz_d;

class Particle {
public:
    double mass, radius;
    double x, y, z; // Position
    double velocityX, velocityY, velocityZ; // Velocity
    double forceX, forceY, forceZ; // Force
    double xi = 0.0; // Thermostat variable
    double xidot = 0.0; // Time derivative of the thermostat variable
    double lastX, lastY, lastZ; // Store the positions at last neighbor list update

    __host__ __device__ void Init(double x0, double y0, double z0, double velocityX0, double velocityY0, double velocityZ0, double mass0, double radius0);
    double GetMass(void);
    double GetX(void);
    double GetY(void);
    double GetZ(void);
    double GetKineticEnergy(void);
    double GetVelocity(void);
    double GetVelocityX(void);
    double GetVelocityY(void);
    double GetVelocityZ(void);
    double GetForceX(void);
    double GetForceY(void);
    double GetForceZ(void);
    double GetNeighborList(void);

    __host__ __device__ void ApplyPeriodicBoundaryConditions(double Lx, double Ly, double Lz);
    __device__ void Move_r(double dt);
    __device__ void Move_V(double dt);

};

__global__ void moveParticlesKernel(Particle* particles, int N, double dt);
__global__ void updateVelocitiesKernel(Particle* particles, int N, double dt);

// Thermostats
__global__ void setupRandomStates(curandState* states, unsigned long seed);
__global__ void applyLangevinThermostat(Particle* particles, int N, double dt, double gamma, double temperature, curandState* states);

#endif
