#include <fstream>
#include <vector>
#include <array>

#include "Particle.h"
#include "system.h"
#include "write_settings.h"

void ComputeVACF(Particle* particles, float* vacf, int N, float time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite) {
    static std::vector<std::array<float, 3>> initialVelocities;
    static bool isInitialized = false;
    float intialVelocityNormalization;

    if (shouldReset || !isInitialized) {
        initialVelocities.clear();
        initialVelocities.resize(N);
        for (int i = 0; i < N; ++i) {
            initialVelocities[i] = {particles[i].GetVelocityX(), particles[i].GetVelocityY(), particles[i].GetVelocityZ()};
        }
        isInitialized = true;
    } 
    
    // Compute VACF and add to existing data
    for (int i = 0; i < N; ++i) {
        float vacfContribution = initialVelocities[i][0] * particles[i].GetVelocityX()
                                + initialVelocities[i][1] * particles[i].GetVelocityY()
                                + initialVelocities[i][2] * particles[i].GetVelocityZ();

        intialVelocityNormalization = initialVelocities[i][0] * initialVelocities[i][0] + 
                                      initialVelocities[i][1] * initialVelocities[i][1] + 
                                      initialVelocities[i][2] * initialVelocities[i][2];
        
        // Accumulate VACF for current count
        vacf[count % maxVACFCount] += vacfContribution / (N * intialVelocityNormalization); // Average over all particles
    }
    
    if (shouldWrite) {
        for (int timeLag = 0; timeLag < maxVACFCount; ++timeLag) {
            outFile << timeLag << "\t" << vacf[timeLag] / (vacfSamplingReps - 1) << std::endl; // Normalize
        }
    }
}