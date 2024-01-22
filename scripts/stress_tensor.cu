#include <cuda_runtime.h>
#include "Particle.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


__global__ void calculateStressTensorCUDA(Particle* particles, double* stressTensor, int N, double volume) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // kinetic contribution - temperature
        for (int alpha = 0; alpha < 3; ++alpha) {
            for (int beta = 0; beta < 3; ++beta) {
                double velocity_alpha = (alpha == 0) ? particles[i].velocityX : (alpha == 1) ? particles[i].velocityY : particles[i].velocityZ;
                double velocity_beta = (beta == 0) ? particles[i].velocityX : (beta == 1) ? particles[i].velocityY : particles[i].velocityZ;

                atomicAdd(&stressTensor[alpha * 3 + beta], particles[i].mass * velocity_alpha * velocity_beta / volume);
            }
        }
        
        // virial part
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;

                for (int alpha = 0; alpha < 3; ++alpha) {
                    for (int beta = 0; beta < 3; ++beta) {
                        double r_alpha = (alpha == 0) ? dx : (alpha == 1) ? dy : dz;
                        double F_beta = (beta == 0) ? particles[i].forceX : (beta == 1) ? particles[i].forceY : particles[i].forceZ;
                        atomicAdd(&stressTensor[alpha * 3 + beta], -r_alpha * F_beta / volume);
                    }
                }
            }
        }
    }
}
