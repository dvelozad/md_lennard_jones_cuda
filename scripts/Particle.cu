#include <cmath>
#include <random>
#include <iostream>

#include "Particle.h"
#include "system.h"

using namespace std;

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


__global__ void moveParticlesKernel(Particle* particles, int N, float dt, float Lx, float Ly, float Lz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles[idx].Move_r(dt, Lx, Ly, Lz);
    }
}

__global__ void updateVelocitiesKernel(Particle* particles, int N, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles[idx].Move_V(dt);
    }
}