#include <cmath>
#include <omp.h>
#include <iostream>

#include "Collider.h"
#include "system.h"
#include "cuda_opt_constants.h"


void Collider::Init(void) {
    potentialEnergy = 0;
}

__global__ void calculateForcesCUDA(Particle* particles, float* partialPotentialEnergy, int N, float Lx, float Ly, float Lz, float epsilon_sigma_6, float sigma_6, float cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float potentialEnergy = 0;

        Particle* pi = &particles[i];
        
        //for (int j = 0; j < N; j++) { // Normal implementation
        // Verlet list
        for (int k = 0; k < particles[i].numNeighbors; k++) {
            int j = particles[i].neighbors[k];

            if(i != j){

            Particle* pj = &particles[j];

            float dx = pi->x - pj->x;
            float dy = pi->y - pj->y;
            float dz = pi->z - pj->z;

            // Apply minimum image convention
            dx -= Lx * round(dx / Lx);
            dy -= Ly * round(dy / Ly);
            dz -= Lz * round(dz / Lz);

            float rsq = dx * dx + dy * dy + dz * dz;
            if (rsq < cutoff * cutoff) {
                float d1 = 1.0 / rsq;
                float d3 = d1 * d1 * d1;
                float forceNormal = 24 * epsilon_sigma_6 * d3 * d1 * ( (2 * sigma_6 * d3) - 1);// - forceNormalCutOff_d * (cutoff / sqrt(rsq));

                atomicAdd(&(pi->forceX), forceNormal * dx);
                atomicAdd(&(pi->forceY), forceNormal * dy);
                atomicAdd(&(pi->forceZ), forceNormal * dz);

                atomicAdd(&(pj->forceX), -forceNormal * dx);
                atomicAdd(&(pj->forceY), -forceNormal * dy);
                atomicAdd(&(pj->forceZ), -forceNormal * dz);

                //float r = sqrt(rsq);
                //potentialEnergy += 4 * epsilon_sigma_6 * d3 * ( (sigma_6 * d3) - 1) - potentialEnergy_cutoff + forceCutoff * (r - cutoff);
            }

            }
        }
    }
}



__global__ void resetForces(Particle* particles, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        particles[i].forceX = 0;
        particles[i].forceY = 0;
        particles[i].forceZ = 0;
    }
}

void Collider::CalculateForces(Particle* dev_particles, float* dev_partialPotentialEnergy, int N, float Lx, float Ly, float Lz, int totalCells, Cell* dev_cells) {
    int blockSize = 128; 
    int numBlocks = (N + blockSize - 1) / blockSize;

    
    resetForces<<<numBlocks, blockSize>>>(dev_particles, N);
    //calculateCellForcesCUDA<<<numBlocks, blockSize>>>(dev_particles, dev_cells, totalCells, Lx, Ly, Lz, epsilon_sigma_6, sigma_6, cutoff);
    calculateForcesCUDA<<<numBlocks, blockSize>>>(dev_particles, dev_partialPotentialEnergy, N, Lx, Ly, Lz, epsilon_sigma_6, sigma_6, cutoff);
}

/*__global__ void calculateForcesCUDA(Particle* particles, float* partialPotentialEnergy, int N, float Lx, float Ly, float Lz, float epsilon_sigma_6, float sigma_6, float cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    extern __shared__ float3 neighborPositions[]; // Declare shared memory for neighbor positions

    float potentialEnergy = 0.0f;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 posI = make_float3(particles[i].x, particles[i].y, particles[i].z);

    int numNeighbors = particles[i].numNeighbors;

    // Prefetch neighbor position into shared memory (if within bounds)
    if (threadIdx.x < blockDim.x) {
        for (int k = 0; k < numNeighbors; k++) {
            if (k < blockDim.x) {
                int neighborIdx = particles[i].neighbors[k];
                neighborPositions[k] = make_float3(particles[neighborIdx].x, particles[neighborIdx].y, particles[neighborIdx].z);
            }
        }
    }
    __syncthreads(); // Synchronize to ensure all positions are loaded

    // Verlet list - force calculation
    for (int k = 0; k < numNeighbors; k++) {
        float3 posJ = neighborPositions[k]; // Use the pre-fetched position from shared memory

        float3 dr = make_float3(posI.x - posJ.x, posI.y - posJ.y, posI.z - posJ.z);

        // Apply minimum image convention
        dr.x -= Lx * round(dr.x / Lx);
        dr.y -= Ly * round(dr.y / Ly);
        dr.z -= Lz * round(dr.z / Lz);

        float rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        if (rsq < cutoff * cutoff && rsq > 1e-12f) { // Avoid division by zero or too close encounters
            float d1 = 1.0f / rsq;
            float d3 = d1 * d1 * d1;
            float forceScalar = 24.0f * epsilon_sigma_6 * d3 * d1 * ((2.0f * sigma_6 * d3) - 1.0f);

            force.x += forceScalar * dr.x;
            force.y += forceScalar * dr.y;
            force.z += forceScalar * dr.z;

            // Potential energy calculation can also be added here if needed
        }
    }

    // Update the global memory with the calculated forces
    particles[i].forceX = force.x;
    particles[i].forceY = force.y;
    particles[i].forceZ = force.z;
    partialPotentialEnergy[i] = potentialEnergy;
}
*/

// Working version - Not optimal
/*__global__ void calculateForcesCUDA(Particle* particles, float* partialPotentialEnergy, int N, float Lx, float Ly, float Lz, float epsilon, float sigma, float cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float potentialEnergy = 0;

        particles[i].forceX = 0;
        particles[i].forceY = 0;
        particles[i].forceZ = 0;

        for (int j = 0; j < N; j++) {
            if (i != j) {
                float dx = particles[i].x - particles[j].x;
                float dy = particles[i].y - particles[j].y;
                float dz = particles[i].z - particles[j].z;

                dx -= Lx * round(dx / Lx);
                dy -= Ly * round(dy / Ly);
                dz -= Lz * round(dz / Lz);

                float distance = sqrt(dx * dx + dy * dy + dz * dz);
                
                if (distance < cutoff) { 
                    // Force at cutoff
                    float forceCutoff = -24*epsilon*((pow(sigma, 6) / pow(cutoff, 7)) - 2*(pow(sigma, 12) / pow(cutoff, 13))); 
                    float potentialEnergyAtCuttoff = 4*epsilon*(pow((sigma/cutoff), 12) - pow((sigma/cutoff), 6));

                    float forceNormal = -24*epsilon*((pow(sigma, 6) / pow(distance, 7)) - 2*(pow(sigma, 12) / pow(distance, 13))) - forceCutoff; 

                    particles[i].forceX += forceNormal * dx / distance;
                    particles[i].forceY += forceNormal * dy / distance;
                    particles[i].forceZ += forceNormal * dz / distance;

                    // Stoddard-Ford linear addition
                    potentialEnergy += 4*epsilon*(pow((sigma/distance), 12) - pow((sigma/distance), 6)) - potentialEnergyAtCuttoff + forceCutoff * (distance - cutoff);
                }
            }
        }
     partialPotentialEnergy[i] = potentialEnergy;
    }
}
*/