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

#include "system.h"
#include "write_settings.h"
#include "cuda_opt_constants.h"


//__device__ double Lx_d, Ly_d, Lz_d;
int main() {

    Readdat();

    cudaMemcpyToSymbol(Lx_d, &L, sizeof(double));
    cudaMemcpyToSymbol(Ly_d, &L, sizeof(double));
    cudaMemcpyToSymbol(Lz_d, &L, sizeof(double));

    cudaMemcpyToSymbol(kB_d, &kB, sizeof(double));
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
    int i, drawTime, currentTimeStep;


    if (L / 2 < cutoff) {
        cerr << "Error : The cutoff distance is greater than half the box dimension L / 2 :" << L / 2 << " - Rc : " << cutoff << endl;
        return 1;
    }

    // To save energies
    std::string temperatureFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_temperature_data.txt";
    std::ofstream outFile_temperature(temperatureFilename);
    if (!outFile_temperature) {
        std::cout << temperatureFilename << endl;
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
    for (currentTimeStep = time = drawTime = 0; currentTimeStep < NumberOfSteps; time += dt, drawTime++) {

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