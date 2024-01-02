#include "Collider.h"
#include "Constants.h"
#include <cmath>
#include <omp.h>
#include <iostream>

// Collider methods
void Collider::Init(void) {
    potentialEnergy = 0;
}

__global__ void calculateForcesCUDA(Particle* particles, double* partialPotentialEnergy, int N, double Lx, double Ly, double Lz, double epsilon, double sigma, double cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double potentialEnergy = 0;

        particles[i].forceX = 0;
        particles[i].forceY = 0;
        particles[i].forceZ = 0;

        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;

                dx -= Lx * round(dx / Lx);
                dy -= Ly * round(dy / Ly);
                dz -= Lz * round(dz / Lz);

                double distance = sqrt(dx * dx + dy * dy + dz * dz);
                
                if (distance < cutoff) { 
                    // Force at cutoff
                    double forceCutoff = -24*epsilon*((pow(sigma, 6) / pow(cutoff, 7)) - 2*(pow(sigma, 12) / pow(cutoff, 13))); 
                    double potentialEnergyAtCuttoff = 4*epsilon*(pow((sigma/cutoff), 12) - pow((sigma/cutoff), 6));

                    double forceNormal = -24*epsilon*((pow(sigma, 6) / pow(distance, 7)) - 2*(pow(sigma, 12) / pow(distance, 13))) - forceCutoff; 

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

void Collider::CalculateForces(Particle* dev_particles, double* dev_partialPotentialEnergy, int N) {
    //std::cout << "Epsilon: " << N << std::endl;

    // grid and block sizes
    int blockSize = 256; 
    int numBlocks = (N + blockSize - 1) / blockSize;

    // CUDA kernel launch
    calculateForcesCUDA<<<numBlocks, blockSize>>>(dev_particles, dev_partialPotentialEnergy, N, Lx, Ly, Lz, epsilon, sigma, cutoff);

    // errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // synchronize device
    cudaDeviceSynchronize();
}
