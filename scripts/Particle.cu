#include <cmath>
#include <random>
#include <iostream>

#include "Particle.h"
#include "system.h"

using namespace std;


////////////////////////////////////////////////////////////////////////////////////
__device__ void atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}
////////////////////////////////////////////////////////////////////////////////////

float Particle::GetMass(void) { return mass; }
float Particle::GetX(void) { return x; }
float Particle::GetY(void) { return y; }
float Particle::GetZ(void) { return z; }
float Particle::GetKineticEnergy(void) { return 0.5*mass * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ); }
float Particle::GetVelocity(void) { return sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ); }
float Particle::GetVelocityX(void) { return velocityX; }
float Particle::GetVelocityY(void) { return velocityY; }
float Particle::GetVelocityZ(void) { return velocityZ; }
float Particle::GetForceX(void) { return forceX; }
float Particle::GetForceY(void) { return forceY; }
float Particle::GetForceZ(void) { return forceZ; }


__host__ void Particle::Init(float x0, float y0, float z0, float velocityX0, float velocityY0, float velocityZ0, float mass0, float radius0) {
    x = x0; y = y0; z = z0;
    velocityX = velocityX0; velocityY = velocityY0; velocityZ = velocityZ0;
    mass = mass0; radius = radius0;

    for(int n = 0; n < MAX_NEIGHBORS; n++){
        neighbors[n] = 0;
    }
}

__device__ void Particle::ApplyPeriodicBoundaryConditions(float Lx, float Ly, float Lz) {
    if (x < 0) x += Lx;
    else if (x > Lx) x -= Lx;

    if (y < 0) y += Ly;
    else if (y > Ly) y -= Ly;

    if (z < 0) z += Lz;
    else if (z > Lz) z -= Lz;
}

__device__ void Particle::Move_r(float dt, float Lx, float Ly, float Lz) {
    x += velocityX * dt;
    y += velocityY * dt;
    z += velocityZ * dt;
    ApplyPeriodicBoundaryConditions(Lx, Ly, Lz);
}

__device__ void Particle::Move_V(float dt) {
    velocityX += (forceX / mass) * dt;
    velocityY += (forceY / mass) * dt;
    velocityZ += (forceZ / mass) * dt;
}


__device__ float calculateDisplacement(const Particle& p, float Lx, float Ly, float Lz) {
    float dx = p.x - p.lastX;
    float dy = p.y - p.lastY;
    float dz = p.z - p.lastZ;

    dx -= Lx * round(dx / Lx);
    dy -= Ly * round(dy / Ly);
    dz -= Lz * round(dz / Lz);

    return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void moveParticlesKernel(Particle* particles, int N, float dt, float Lx, float Ly, float Lz, float* dev_maxDisplacement) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles[idx].Move_r(dt, Lx, Ly, Lz);

        // Calculate displacement
        float displacement = calculateDisplacement(particles[idx], Lx, Ly, Lz);
        atomicMax(dev_maxDisplacement, displacement);
    }
}

__global__ void updateVelocitiesKernel(Particle* particles, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles[idx].Move_V(dt);
    }
}
