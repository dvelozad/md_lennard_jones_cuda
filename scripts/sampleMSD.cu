#include <fstream>
#include <vector>
#include <array>

#include "Particle.h"
#include "write_settings.h"

void ComputeMSD(Particle* particles, double* msd, int N, double time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite) {
    static std::vector<std::array<double, 3>> initialPositions;
    static bool isInitialized = false;

    if (shouldReset || !isInitialized) {
        initialPositions.clear();
        initialPositions.resize(N);
        for (int i = 0; i < N; ++i) {
            initialPositions[i] = {particles[i].GetX(), particles[i].GetY(), particles[i].GetZ()};
        }
        isInitialized = true;
    }

    // Compute MSD and add to existing data
    for (int i = 0; i < N; ++i) {
        double dx = particles[i].GetX() - initialPositions[i][0];
        double dy = particles[i].GetY() - initialPositions[i][1];
        double dz = particles[i].GetZ() - initialPositions[i][2];
        double displacementSquared = dx*dx + dy*dy + dz*dz;

        msd[count % maxMSDCount] += displacementSquared / N; // Average over all particles
    }

    if (shouldWrite) {
        // Normalize and output MSD
        for (int timeLag = 0; timeLag < maxMSDCount; ++timeLag) {
            outFile << timeLag << "\t" << msd[timeLag] / (msdSamplingReps - 1) << std::endl;
        }
    }
}