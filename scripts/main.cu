// main.cpp
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "Random64.h"
#include "Particle.h"
#include "Collider.h"
#include "Constants.h"


using namespace std;


double CalculateCurrentTemperature(Particle *particles, int numParticles) {
    double totalKineticEnergy = 0.0;

    for (int i = 0; i < numParticles; ++i) {
        totalKineticEnergy += particles[i].GetKineticEnergy();
    }

    double temperature = (2.0 / (3.0 * numParticles * kB)) * totalKineticEnergy;
    return temperature;
}

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

__global__ void applyLangevinThermostat(Particle* particles, int N, double dt, double gamma, double temperature, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Generate a normally distributed random number
        double randX = curand_normal_double(&states[idx]);
        double randY = curand_normal_double(&states[idx]);
        double randZ = curand_normal_double(&states[idx]);

        // Calculate the random force magnitude
        double randomForceMagnitude = sqrt(2.0 * particles[idx].mass * kB * temperature * gamma * dt);

        // Update velocity
        particles[idx].velocityX += (particles[idx].forceX / particles[idx].mass - gamma * particles[idx].velocityX) * dt 
                                   + randomForceMagnitude * randX / particles[idx].mass;
        particles[idx].velocityY += (particles[idx].forceY / particles[idx].mass - gamma * particles[idx].velocityY) * dt 
                                   + randomForceMagnitude * randY / particles[idx].mass;
        particles[idx].velocityZ += (particles[idx].forceZ / particles[idx].mass - gamma * particles[idx].velocityZ) * dt 
                                   + randomForceMagnitude * randZ / particles[idx].mass;
    }
}

__global__ void setupRandomStates(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}



//__device__ double Lx_d, Ly_d, Lz_d;
int main() {
    cudaMemcpyToSymbol(Lx_d, &L, sizeof(double));
    cudaMemcpyToSymbol(Ly_d, &L, sizeof(double));
    cudaMemcpyToSymbol(Lz_d, &L, sizeof(double));


    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Simulation label : " + simulationLabel << endl;
    std::cout << "Box dimension L : " << Lx << endl;
    std::cout << "Number of particles : " << N << endl;

    Particle particles[N];
    Collider collider;
    Crandom randomGenerator(0);
    double time, drawTime, radius, kineticEnergy, potentialEnergy, T_current;
    int i;


    if (L / 2 < cutoff) {
        cerr << "Error : The cutoff distance is greater than half the box dimension L / 2 :" << L / 2 << " - Rc : " << cutoff << endl;
        return 1;
    }

    // To save energies
    std::string energyFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_energy_data.txt";
    std::ofstream outFile_energy(energyFilename);
    if (!outFile_energy) {
        cerr << "Error opening file for writing" << endl;
        return 1;
    }

    // To save energies
    std::string temperatureFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_temperature_data.txt";
    std::ofstream outFile_temperature(temperatureFilename);
    if (!outFile_temperature) {
        cerr << "Error opening file for writing" << endl;
        return 1;
    }

    // To save positions
    std::string positionsFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_positions_data.txt";
    std::ofstream outFile_positions(positionsFilename);
    if (!outFile_positions) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    // To save positions
    std::string velocitiesFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_velocities_data.txt";
    std::ofstream outFile_velocities(velocitiesFilename);
    if (!outFile_velocities) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    // To save RDF
    std::string outFile_RDF = "../output_files/" + simulationLabel + '/' + simulationLabel + "rdf_results.txt";

    // Intit collider
    collider.Init();

    int unitCellsPerSide = std::cbrt(N / 4);
    double a = Lx / unitCellsPerSide;

    std::vector<std::array<double, 3>> velocities(N);
    std::vector<std::array<double, 3>> positions(N); // Store initial positions

    // Assign random velocities and positions
    double totalVx = 0, totalVy = 0, totalVz = 0;
    int particleIndex = 0;
    for (int ix = 0; ix < unitCellsPerSide; ix++) {
        for (int iy = 0; iy < unitCellsPerSide; iy++) {
            for (int iz = 0; iz < unitCellsPerSide; iz++) {
                std::vector<std::array<double, 3>> unitCellPositions = {
                    {ix * a, iy * a, iz * a},
                    {(ix + 0.5) * a, (iy + 0.5) * a, iz * a},
                    {ix * a, (iy + 0.5) * a, (iz + 0.5) * a},
                    {(ix + 0.5) * a, iy * a, (iz + 0.5) * a}
                };

                for (auto& pos : unitCellPositions) {
                    if (particleIndex < N) {
                        double theta = 2 * M_PI * randomGenerator.r();
                        double phi = acos(2 * randomGenerator.r() - 1);
                        double randomInitialVelocity = randomGenerator.r() * InitialVelocity;

                        double velocityX0 = randomInitialVelocity * sin(phi) * cos(theta);
                        double velocityY0 = randomInitialVelocity * sin(phi) * sin(theta);
                        double velocityZ0 = randomInitialVelocity * cos(phi);

                        velocities[particleIndex] = {velocityX0, velocityY0, velocityZ0};
                        positions[particleIndex] = pos;

                        totalVx += velocityX0;
                        totalVy += velocityY0;
                        totalVz += velocityZ0;

                        particleIndex++;
                    }
                }
            }
        }
    }

    // Adjust velocities to ensure zero average
    double avgVx = totalVx / N;
    double avgVy = totalVy / N;
    double avgVz = totalVz / N;

    for (int i = 0; i < N; i++) {
        particles[i].Init(
            positions[i][0], positions[i][1], positions[i][2],
            velocities[i][0] - avgVx, velocities[i][1] - avgVy, velocities[i][2] - avgVz,
            defaultMass, radius);
    }


    Particle* dev_particles;
    double* dev_partialPotentialEnergy;
    cudaMalloc(&dev_particles, N * sizeof(Particle));
    cudaMalloc(&dev_partialPotentialEnergy, N * sizeof(double));
    cudaMemcpy(dev_particles, particles, N * sizeof(Particle), cudaMemcpyHostToDevice);


    // kernel config
    int blockSize = 256; 
    int numBlocks = (N + blockSize - 1) / blockSize;

    // random states
    curandState* devStates;
    cudaMalloc(&devStates, N * sizeof(curandState));
    setupRandomStates<<<numBlocks, blockSize>>>(devStates, 12472547);

    collider.CalculateForces(dev_particles, dev_partialPotentialEnergy, N);
    for (time = drawTime = 0; time < totalTime; time += dt, drawTime += dt) {

        // Write info
        if (int(time / timeFrame) != int((time - dt) / timeFrame)) {

            // Copy data back to host if needed
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);

            for (i = 0; i < N; i++){
                outFile_positions << i << " " << time << " " << particles[i].GetX() << " " << particles[i].GetY() << " " << particles[i].GetZ() << endl;
                outFile_velocities << i << " " << time << " " << particles[i].GetVelocityX() << " " << particles[i].GetVelocityY() << " " << particles[i].GetVelocityZ() << endl;
            }

            double* partialPotentialEnergy = new double[N];
            cudaMemcpy(partialPotentialEnergy, dev_partialPotentialEnergy, N * sizeof(double), cudaMemcpyDeviceToHost);
            potentialEnergy = 0;
            for (int i = 0; i < N; i++) {
                potentialEnergy += partialPotentialEnergy[i];
            }
            delete[] partialPotentialEnergy;

            // Get energy
            kineticEnergy = 0;
            for (int i = 0; i < N; i++) kineticEnergy += particles[i].GetKineticEnergy();

            // Get temperature
            T_current = CalculateCurrentTemperature(particles, N);

            // Write
            outFile_energy << time << " " << kineticEnergy << " " << potentialEnergy << endl;
            outFile_temperature << time << " " << T_current << endl;


        }  

        // call kernels to update particle velocities and positions
        updateVelocitiesKernel<<<numBlocks, blockSize>>>(dev_particles, N,  dt * 0.5);
        applyLangevinThermostat<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5, Gamma, T_desired, devStates);

        moveParticlesKernel<<<numBlocks, blockSize>>>(dev_particles, N,  dt);


        // calcu forces
        collider.CalculateForces(dev_particles, dev_partialPotentialEnergy, N);

        // half-step velocity update
        updateVelocitiesKernel<<<numBlocks, blockSize>>>(dev_particles, N,  dt * 0.5);
        applyLangevinThermostat<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5, Gamma, T_desired, devStates);
    }

    //cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(dev_particles);

    return 0;
}