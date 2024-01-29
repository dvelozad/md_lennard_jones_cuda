#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "system.h"
#include "cuda_opt_constants.h"
#include "Particle.h"

// For the langevin thermostat
__global__ void setupRandomStates(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

float CalculateCurrentTemperature(Particle *particles, int numParticles) {
    float totalKineticEnergy = 0.0;

    for (int i = 0; i < numParticles; ++i) {
        totalKineticEnergy += particles[i].GetKineticEnergy();
    }

    float temperature = (2.0 / (3.0 * numParticles * kB)) * totalKineticEnergy;
    return temperature;
}

__global__ void applyLangevinThermostat(Particle* particles, int N, float dt, float gamma, float temperature, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Generate a normally distributed random number
        float randX = curand_normal_double(&states[idx]);
        float randY = curand_normal_double(&states[idx]);
        float randZ = curand_normal_double(&states[idx]);

        // Calculate the random force magnitude
        float randomForceMagnitude = sqrt(2.0 * particles[idx].mass * kB_d * temperature * gamma * dt);

        // Update velocity
        particles[idx].velocityX += (particles[idx].forceX / particles[idx].mass - gamma * particles[idx].velocityX) * dt 
                                   + randomForceMagnitude * randX / particles[idx].mass;
        particles[idx].velocityY += (particles[idx].forceY / particles[idx].mass - gamma * particles[idx].velocityY) * dt 
                                   + randomForceMagnitude * randY / particles[idx].mass;
        particles[idx].velocityZ += (particles[idx].forceZ / particles[idx].mass - gamma * particles[idx].velocityZ) * dt 
                                   + randomForceMagnitude * randZ / particles[idx].mass;
    }
}