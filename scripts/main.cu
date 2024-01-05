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


void ComputeVACF(Particle* particles, double* vacf, int N, double time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite) {
    static std::vector<std::array<double, 3>> initialVelocities;
    static bool isInitialized = false;
    double intialVelocityNormalization;

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
        double vacfContribution = initialVelocities[i][0] * particles[i].GetVelocityX()
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



__global__ void computeDistancesCUDA(double* x, double* y, double* z, double* distances, int N, double Lx, double Ly, double Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N && i != j) {
        double dx = x[i] - x[j];
        double dy = y[i] - y[j];
        double dz = z[i] - z[j];

        dx -= Lx * round(dx / Lx);
        dy -= Ly * round(dy / Ly);
        dz -= Lz * round(dz / Lz);

        double distance = sqrt(dx * dx + dy * dy + dz * dz);
        distances[i * N + j] = distance;
    }
}


void computeRDFCUDA(Particle* particles, int N, double Lx, double Ly, double Lz, double maxDistance, int numBins, double time, std::ofstream& outFile) {
    // Allocate memory on GPU
    //cout << maxDistance << endl;
    double *dev_x, *dev_y, *dev_z, *dev_distances;
    cudaMalloc(&dev_x, N * sizeof(double));
    cudaMalloc(&dev_y, N * sizeof(double));
    cudaMalloc(&dev_z, N * sizeof(double));
    cudaMalloc(&dev_distances, N * N * sizeof(double));

    // Copy data to GPU
    std::vector<double> x(N), y(N), z(N);
    for (int i = 0; i < N; ++i) {
        x[i] = particles[i].GetX();
        y[i] = particles[i].GetY();
        z[i] = particles[i].GetZ();
    }
    cudaMemcpy(dev_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, z.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(32, 32); 
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch CUDA kernel
    computeDistancesCUDA<<<gridSize, blockSize>>>(dev_x, dev_y, dev_z, dev_distances, N, Lx, Ly, Lz);

    // Copy distances back to CPU
    std::vector<double> distances(N * N);
    cudaMemcpy(distances.data(), dev_distances, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    cudaFree(dev_distances);

    // Histogram the distances and compute RDF
    std::vector<int> bins(numBins, 0);
    double binSize = maxDistance / numBins;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j && distances[i * N + j] < maxDistance) {
                int binIndex = distances[i * N + j] / binSize;
                if (binIndex < numBins) {
                    bins[binIndex]++;
                }
            }
        }
    }

    // Normalize RDF
    double density = N / (Lx * Ly * Lz);
    std::vector<double> rdf(numBins);
    //std::ofstream outFile(outputFilename);
    for (int i = 0; i < numBins; ++i) {
        double r1 = i * binSize;
        double r2 = r1 + binSize;
        double shellVolume = (4.0 / 3.0) * M_PI * (r2 * r2 * r2 - r1 * r1 * r1);
        rdf[i] = bins[i] / (shellVolume * density * N);
        outFile << time << "\t" << (r1 + r2) / 2 << "\t" << rdf[i] << endl;
    }
    //outFile.close();
}


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

// For the langevin thermostat
__global__ void setupRandomStates(curandState* states, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}


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


//__device__ double Lx_d, Ly_d, Lz_d;
int main() {
    cudaMemcpyToSymbol(Lx_d, &L, sizeof(double));
    cudaMemcpyToSymbol(Ly_d, &L, sizeof(double));
    cudaMemcpyToSymbol(Lz_d, &L, sizeof(double));

    cudaMemcpyToSymbol(epsilon_sigma_6_d, &epsilon_sigma_6, sizeof(double));
    cudaMemcpyToSymbol(sigma_6_d, &sigma_6, sizeof(double));
    cudaMemcpyToSymbol(forceNormalCutOff_d, &forceNormalCutOff, sizeof(double));

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
    double time, radius, kineticEnergy, potentialEnergy, T_current;
    int i, drawTime;


    if (L / 2 < cutoff) {
        cerr << "Error : The cutoff distance is greater than half the box dimension L / 2 :" << L / 2 << " - Rc : " << cutoff << endl;
        return 1;
    }

    // To save energies
    std::string temperatureFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_temperature_data.txt";
    std::ofstream outFile_temperature(temperatureFilename);
    if (!outFile_temperature) {
        cerr << "Error opening file for writing" << endl;
        return 1;
    }

    /*   
    // To save energies
    std::string energyFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_energy_data.txt";
    std::ofstream outFile_energy(energyFilename);
    if (!outFile_energy) {
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
    */

    // To save positions
    std::string rdfFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_rdf_data.txt";
    std::ofstream outFile_rdf(rdfFilename);
    if (!outFile_rdf) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    std::string vafFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_vaf_data.txt";
    std::ofstream outFile_vacf(vafFilename);
    if (!outFile_vacf) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    std::string msdFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_msd_data.txt";
    std::ofstream outFile_msd(msdFilename);
    if (!outFile_msd) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    /*    // To save positions
    std::string velocitiesFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_velocities_data.txt";
    std::ofstream outFile_velocities(velocitiesFilename);
    if (!outFile_velocities) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    // To save positions
    std::string forcesFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_forces_data.txt";
    std::ofstream outFile_forces(forcesFilename);
    if (!outFile_forces) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }

    // To save positions
    std::string stressFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_stress_tensor_data.txt";
    std::ofstream outFile_stress(stressFilename);
    if (!outFile_stress) {
        cerr << "Error opening file for writing" << endl;
        return 1; 
    }*/

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


    // Allocate memory on GPU
    double* dev_stressTensor;
    cudaMalloc(&dev_stressTensor, 9 * sizeof(double)); // 3x3 stress tensor
    cudaMemset(dev_stressTensor, 0, 9 * sizeof(double));


    double boxVolume = L*L*L;
    //calculateStressTensorCUDA<<<numBlocks, blockSize>>>(dev_particles, dev_stressTensor, N, boxVolume);

    // Copy stress tensor back to host
    double stressTensor[9];
    //cudaMemcpy(stressTensor, dev_stressTensor, 9 * sizeof(double), cudaMemcpyDeviceToHost);


    //////////////////////////////////////////////////////////////////////////////
    //// Equilibartion
    //////////////////////////////////////////////////////////////////////////////
    collider.CalculateForces(dev_particles, dev_partialPotentialEnergy, N);
    for (int eqStep = 0; eqStep < equilibrationSteps; eqStep++) {

        if(eqStep % eqVerboseFrame == 0){
            cout << "Equilibration step : " << eqStep << endl;
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


    cout << "-----------------------------------------------------" << endl;


    std::vector<double> vacf(maxVACFCount, 0.0);
    int vacfCount = 0;
    int vacfSamplingCount = 0;

    std::vector<double> msd(maxMSDCount, 0.0);
    int msdCount = 0;
    int msdSamplingCount = 0;
    for (time = drawTime = 0; time < totalTime; time += dt, drawTime++) {

        /* 
        if (rawTime % timeFrame == 0){     
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);

            for (i = 0; i < N; i++){
                outFile_positions << i << " " << time << " " << particles[i].GetX() << " " << particles[i].GetY() << " " << particles[i].GetZ() << endl;
                outFile_velocities << i << " " << time << " " << particles[i].GetVelocityX() << " " << particles[i].GetVelocityY() << " " << particles[i].GetVelocityZ() << endl;
                outFile_forces << i << " " << time << " " << particles[i].GetForceX() << " " << particles[i].GetForceY() << " " << particles[i].GetForceZ() << endl;
            }
        } 
        */

        if (drawTime % vacf_writeFrame == 0 && vacfSamplingCount < vacfSamplingReps) {
            bool shouldWrite = (vacfSamplingCount == vacfSamplingReps - 1 && (vacfCount % (maxVACFCount - 1)) == 0);
            bool shouldReset = (vacfCount % maxVACFCount) == 0;

            if (shouldReset) {
                cout << "Resetting VACF sampling at time: " << time << endl;
                vacfSamplingCount++;
            }

            cout << "Computing VACF for time: " << time << endl;
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            ComputeVACF(particles, vacf.data(), N, time, vacfCount, outFile_vacf, shouldReset, shouldWrite);
            vacfCount++;
        }

        if (drawTime % msd_writeFrame == 0 && msdSamplingCount < msdSamplingReps) {
            bool shouldWrite = (msdSamplingCount == msdSamplingReps - 1 && (msdCount % (maxMSDCount - 1)) == 0);
            bool shouldReset = (msdCount % maxMSDCount) == 0;

            if (shouldReset) {
                cout << "Resetting MSD sampling at time: " << time << endl;
                msdSamplingCount++;
            }

            cout << "Computing MSD for time: " << time << endl;
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            ComputeMSD(particles, msd.data(), N, time, msdCount, outFile_msd, shouldReset, shouldWrite);
            msdCount++;
        }


       if (drawTime % RDF_writeFrame == 0){
            cout << "Writting RDF for time : " << time << endl;
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            computeRDFCUDA(particles, N, Lx, Ly, Lz, Lx / 2, numBins, time, outFile_rdf);
       }

        if (drawTime % temperature_writeFrame == 0){
            cout << "Writting temperature for time : " << time << endl;
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
            T_current = CalculateCurrentTemperature(particles, N);
            outFile_temperature << time << " " << T_current << endl;
       }


        /* // Write info
        if (drawTime % timeFrame == 0) {

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
            //outFile_energy << time << " " << kineticEnergy << " " << potentialEnergy << endl;
            outFile_temperature << time << " " << T_current << endl;

            // Assuming the particles array is filled with Particle objects


            // stress tensor cal
            //calculateStressTensorCUDA<<<numBlocks, blockSize>>>(dev_particles, dev_stressTensor, N, boxVolume);
            //cudaMemcpy(stressTensor, dev_stressTensor, 9 * sizeof(double), cudaMemcpyDeviceToHost);
            //outFile_stress << time << " " << stressTensor[0] << " " << stressTensor[1] << " " << stressTensor[2] << " " << stressTensor[3] << " " << stressTensor[4] << " " << stressTensor[5] << " " << stressTensor[6] << " " << stressTensor[7] << " " << stressTensor[8] << endl;

        }  */

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