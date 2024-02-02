#ifndef PARTICLE_H
#define PARTICLE_H

#include "system.h"
#include <curand_kernel.h>


struct Cell {
    int particleIndices[MAX_PARTICLES_PER_CELL];
    int numParticles = 0;
    bool overflow;
    int3 stencil[MAX_STENCIL_SIZE]; 
    int3 halfstencil[MAX_STENCIL_SIZE]; 
    int stencilSize = 0;
};

struct Particle {
    // Basic properties
    float mass, radius;
    float x, y, z;                          // Position
    float velocityX, velocityY, velocityZ;  // Velocity
    float forceX, forceY, forceZ;           // Force
    float lastX, lastY, lastZ;              // last neighbor list update

    int neighbors[MAX_NEIGHBORS];
    int numNeighbors = 0;
    int cellIndex = 0;
    
    // CUDA kernels
    __host__  void Init(float x0, float y0, float z0, float velocityX0, float velocityY0, float velocityZ0, float mass0, float radius0);
    __device__ void ApplyPeriodicBoundaryConditions(float Lx, float Ly, float Lz);
    __device__ void Move_r(float dt, float Lx, float Ly, float Lz);
    __device__ void Move_V(float dt);

    // Info
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

};

// Move particles kernels
__global__ void moveParticlesKernel(Particle* particles, int N, float dt, float Lx, float Ly, float Lz, float* dev_maxDisplacement);
__global__ void updateVelocitiesKernel(Particle* particles, int N, float dt);

// Thermostats
__global__ void setupRandomStates(curandState* states, unsigned long seed);
__global__ void applyLangevinThermostat(Particle* particles, int N, float dt, float kB, float gamma, float temperature, curandState* states);

// Neighbor related kernels
__global__ void updateVerletListKernel(Particle* particles, Cell* cells, int N, float cutoff_, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz);

// Cell list realted kernels
__global__ void resetCells(Cell* cells, int totalNumCells);
__global__ void assignParticlesToCells(Particle* particles, Cell* cells, int N, float cellSize, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz);

//// ** produce cell stencil ** 
__global__ void generateStencils(Cell* cells, int numCellsX, int numCellsY, int numCellsZ, float cellLength, float cutoff);

#endif
