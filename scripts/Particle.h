#ifndef PARTICLE_H
#define PARTICLE_H

#include "system.h"
#include <curand_kernel.h>

class Particle {
public:
    float mass, radius;
    float x, y, z; // Position
    float velocityX, velocityY, velocityZ; // Velocity
    float forceX, forceY, forceZ; // Force
    float xi = 0.0; // Thermostat variable
    float xidot = 0.0; // Time derivative of the thermostat variable
    float lastX, lastY, lastZ; // Store the positions at last neighbor list update

    __host__  void Init(float x0, float y0, float z0, float velocityX0, float velocityY0, float velocityZ0, float mass0, float radius0);
    float GetMass(void);
    float GetX(void);
    float GetY(void);
    float GetZ(void);
    float GetKineticEnergy(void);
    float GetVelocity(void);
    float GetVelocityX(void);
    float GetVelocityY(void);
    float GetVelocityZ(void);
    float GetForceX(void);
    float GetForceY(void);
    float GetForceZ(void);
    float GetNeighborList(void);

    __device__ void ApplyPeriodicBoundaryConditions(float Lx, float Ly, float Lz);
    __device__ void Move_r(float dt, float Lx, float Ly, float Lz);
    __device__ void Move_V(float dt);

};

__global__ void moveParticlesKernel(Particle* particles, int N, float dt, float Lx, float Ly, float Lz);
__global__ void updateVelocitiesKernel(Particle* particles, int N, float dt);

// Thermostats
__global__ void setupRandomStates(curandState* states, unsigned long seed);
__global__ void applyLangevinThermostat(Particle* particles, int N, float dt, float gamma, float temperature, curandState* states);

#endif
