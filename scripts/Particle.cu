#include <cmath>
#include <random>
#include <iostream>

#include "Particle.h"
#include "system.h"

using namespace std;

__global__ void moveParticlesKernel(Particle* particles, int N, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles[idx].Move_r(dt);
    }
}

__global__ void updateVelocitiesKernel(Particle* particles, int N, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        particles[idx].Move_V(dt);
    }
}

__host__ __device__ void Particle::Init(double x0, double y0, double z0, double velocityX0, double velocityY0, double velocityZ0, double mass0, double radius0) {
    x = x0; y = y0; z = z0;
    velocityX = velocityX0; velocityY = velocityY0; velocityZ = velocityZ0;
    mass = mass0; radius = radius0;
}

__host__ __device__ void Particle::ApplyPeriodicBoundaryConditions(double Lx, double Ly, double Lz) {
    if (x < 0) x += Lx;
    else if (x > Lx) x -= Lx;

    if (y < 0) y += Ly;
    else if (y > Ly) y -= Ly;

    if (z < 0) z += Lz;
    else if (z > Lz) z -= Lz;
}

__device__ void Particle::Move_r(double dt) {
    x += velocityX * dt;
    y += velocityY * dt;
    z += velocityZ * dt;
    ApplyPeriodicBoundaryConditions(Lx_d, Ly_d, Lz_d);
}

__device__ void Particle::Move_V(double dt) {
    velocityX += (forceX / mass) * dt;
    velocityY += (forceY / mass) * dt;
    velocityZ += (forceZ / mass) * dt;
}

double Particle::GetMass(void) { return mass; }
double Particle::GetX(void) { return x; }
double Particle::GetY(void) { return y; }
double Particle::GetZ(void) { return z; }
double Particle::GetKineticEnergy(void) { return 0.5*mass * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ); }
double Particle::GetVelocity(void) { return sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ); }
double Particle::GetVelocityX(void) { return velocityX; }
double Particle::GetVelocityY(void) { return velocityY; }
double Particle::GetVelocityZ(void) { return velocityZ; }
double Particle::GetForceX(void) { return forceX; }
double Particle::GetForceY(void) { return forceY; }
double Particle::GetForceZ(void) { return forceZ; }