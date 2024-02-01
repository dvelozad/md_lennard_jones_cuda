#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "Random64.h"
#include "Particle.h"
#include "Collider.h"

#include "system.h"
#include "write_settings.h"
#include "cuda_opt_constants.h"


//__device__ float Lx_d, Ly_d, Lz_d;
int main() {
    Readdat();

    OutputManager outputManager;
    outputManager.setOutputNames(simulationLabel);
    outputManager.openFiles();

    std::ofstream& outFile_positions  = outputManager.getPosFile();
    std::ofstream& outFile_velocities = outputManager.getVelFile();
    std::ofstream& outFile_forces     = outputManager.getForcesFile();

    std::ofstream& outFile_rdf  = outputManager.getRdfFile();
    std::ofstream& outFile_vacf = outputManager.getVafFile();
    std::ofstream& outFile_msd  = outputManager.getMsdFile();

    std::ofstream& outFile_temperature = outputManager.getTemperatureFile();

    cudaMemcpyToSymbol(kB_d, &kB, sizeof(float));
    cudaMemcpyToSymbol(epsilon_sigma_6_d, &epsilon_sigma_6, sizeof(float));
    cudaMemcpyToSymbol(sigma_6_d, &sigma_6, sizeof(float));
    cudaMemcpyToSymbol(forceNormalCutOff_d, &forceNormalCutOff, sizeof(float));


    cout << "dt : " << dt << endl;
    cout << "equilibrationSteps : " << equilibrationSteps << endl;
    cout << "NumberOfSteps : " << NumberOfSteps << endl;

    cout << "kB : " << kB << endl;
    cout << "epsilon : " << epsilon << endl;
    cout << "sigma : " << sigma << endl;
    cout << "cutoff : " << cutoff << endl;

    cout << "Gamma : " << Gamma << endl;
    cout << "T_desired : " << T_desired << endl;

    cout << "defaultMass : " << defaultMass << endl;
    cout << "InitialVelocity : " << InitialVelocity << endl;

    cout << "RHO : " << RHO << endl;
    cout << "L : " << L << endl;

    cout << "sigma_6 : " << sigma_6 << endl;
    cout << "forceNormalCutOff : " << forceNormalCutOff << endl;
    cout << "epsilon_sigma_6 : " << epsilon_sigma_6 << endl;
    cout << "potentialEnergy_cutoff : " << potentialEnergy_cutoff << endl;


    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    Particle particles[N];
    Collider collider;
    Crandom randomGenerator(0);
    float time, radius, kineticEnergy, potentialEnergy, T_current;
    int i, drawTime, currentTimeStep;


    if (L / 2 < cutoff) {
        cerr << "Error : The cutoff distance is greater than half the box dimension L / 2 :" << L / 2 << " - Rc : " << cutoff << endl;
        return 1;
    }


    // Intit collider
    collider.Init();

    int unitCellsPerSide = std::cbrt(N / 4);
    float a = Lx / unitCellsPerSide;

    std::vector<std::array<double, 3>> velocities(N);
    std::vector<std::array<double, 3>> positions(N); // Store initial positions

    // Assign random velocities and positions
    float totalVx = 0, totalVy = 0, totalVz = 0;
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
                        float theta = 2 * M_PI * randomGenerator.r();
                        float phi = acos(2 * randomGenerator.r() - 1);
                        float randomInitialVelocity = randomGenerator.r() * InitialVelocity;

                        float velocityX0 = randomInitialVelocity * sin(phi) * cos(theta);
                        float velocityY0 = randomInitialVelocity * sin(phi) * sin(theta);
                        float velocityZ0 = randomInitialVelocity * cos(phi);

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
    float avgVx = totalVx / N;
    float avgVy = totalVy / N;
    float avgVz = totalVz / N;

    for (int i = 0; i < N; i++) {
        particles[i].Init(
            positions[i][0], positions[i][1], positions[i][2],
            velocities[i][0] - avgVx, velocities[i][1] - avgVy, velocities[i][2] - avgVz,
            defaultMass, radius);
    }


    Particle* dev_particles;
    float* dev_partialPotentialEnergy;

    
    cudaMalloc(&dev_particles, N * sizeof(Particle));
    cudaMalloc(&dev_partialPotentialEnergy, N * sizeof(float));


    cudaMemcpy(dev_particles, particles, N * sizeof(Particle), cudaMemcpyHostToDevice);


    // kernel config
    int blockSize = 256; 
    int numBlocks = (N + blockSize - 1) / blockSize;

    // random states
    curandState* devStates;
    cudaMalloc(&devStates, N * sizeof(curandState));
    setupRandomStates<<<numBlocks, blockSize>>>(devStates, 12472547);


    // Allocate memory on GPU
    float* dev_stressTensor;
    cudaMalloc(&dev_stressTensor, 9 * sizeof(float)); // 3x3 stress tensor
    cudaMemset(dev_stressTensor, 0, 9 * sizeof(float));


    float boxVolume = L*L*L;
    //calculateStressTensorCUDA<<<numBlocks, blockSize>>>(dev_particles, dev_stressTensor, N, boxVolume);

    // Copy stress tensor back to host
    float stressTensor[9];
    //cudaMemcpy(stressTensor, dev_stressTensor, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    //////////////////////////////////////////////////////////////////////////////
    //// Neighbor list
    //////////////////////////////////////////////////////////////////////////////
    float displacementThreshold = displacementProportion;

    // Copy the result back to the host
    float maxDisplacement;
    float* dev_maxDisplacement;
    cudaMalloc((void**)&dev_maxDisplacement, sizeof(float));

    // Initialize maxDisplacement to 0
    float zero = 0.0f;
    cudaMemcpy(dev_maxDisplacement, &zero, sizeof(float), cudaMemcpyHostToDevice);


    float interactionCutoff = Lx; 

    int numCellsX = static_cast<int>(floor(Lx / interactionCutoff));
    int numCellsY = static_cast<int>(floor(Ly / interactionCutoff));
    int numCellsZ = static_cast<int>(floor(Lz / interactionCutoff));

    float cellSize = Lx / numCellsX;

    cout << "cellSize : " << cellSize << " " << cellSize *  numCellsX << endl;

/*    int numCellsX = 3;
    int numCellsY = 3;
    int numCellsZ = 3;*/

    
    
    int totalNumCells = numCellsX * numCellsY * numCellsZ;
    Cell cells[totalNumCells];

    Cell* dev_cells;
    cudaMalloc(&dev_cells, totalNumCells * sizeof(Cell));


    //////////////////////////////////////////////////////////////////////////////
    //// Equilibartion
    //////////////////////////////////////////////////////////////////////////////
    generateStencils<<<numBlocks, blockSize>>>(dev_cells, numCellsX, numCellsY, numCellsZ, cellSize, cutoff);

    resetCells<<<numBlocks, blockSize>>>(dev_cells, totalNumCells);
    assignParticlesToCells<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, cellSize, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
    
    updateVerletListKernel<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, extendedCutoff, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);



    cudaMemcpy(cells, dev_cells, totalNumCells * sizeof(Cell), cudaMemcpyDeviceToHost);
    cout << " Cell dim : " << numCellsX << " " << numCellsY << " " << numCellsZ << " "<< endl;
    cout << " Number of particles per cell : " << cells[0].numParticles << endl;
    for(int mm = 0; mm < totalNumCells; mm++){
        cout << mm << " ----------------------------" << endl;
        for(int nn = 0; nn < MAX_PARTICLES_PER_CELL; nn++){
            cout << cells[mm].particleIndices[nn] << " ";
        }
        cout << endl;
    }

    cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    for(int mm = 0; mm < 10; mm++){
        cout << " Number of particles per particle : " << particles[mm].numNeighbors << endl;
        cout << mm << " ----------------------------" << endl;
        for(int nn = 0; nn < particles[mm].numNeighbors; nn++){
            cout << particles[mm].neighbors[nn] << " ";
        }
        cout << endl;
    }

    collider.CalculateForces(dev_particles, dev_partialPotentialEnergy, N, Lx, Ly, Lz, totalNumCells, dev_cells);
    for (int eqStep = 0; eqStep < equilibrationSteps; eqStep++) {
        if(eqStep % eqVerboseFrame == 0){
            cout << "Equilibration step : " << eqStep << endl;
        }

        // Update particle velocities (half-step)
        updateVelocitiesKernel<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5);

        // Move particles and calculate displacements
        moveParticlesKernel<<<numBlocks, blockSize>>>(dev_particles, N, dt, Lx, Ly, Lz, dev_maxDisplacement);
        cudaMemcpy(&maxDisplacement, dev_maxDisplacement, sizeof(float), cudaMemcpyDeviceToHost);

        resetCells<<<numBlocks, blockSize>>>(dev_cells, totalNumCells);
        cudaDeviceSynchronize();

        // Assign particles to cells
        assignParticlesToCells<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, cellSize, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
        //identifyNearestNeighborCells<<<numBlocks, blockSize>>>(dev_particles, N, cellSize, numCellsX, numCellsY, numCellsZ, extendedCutoff);
        //cout << maxDisplacement << endl;

        // Update the Verlet list if necessary
        if (maxDisplacement > displacementThreshold) {
            //cout << "-----------------------------------------------------" << endl;
            //updateVerletListKernel<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, extendedCutoff, cellSize, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
            updateVerletListKernel<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, extendedCutoff, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
            cudaMemset(dev_maxDisplacement, 0, sizeof(float)); // Reset maxDisplacement on the device
            //maxDisplacement = 0 ;
        }

        // Calculate forces
        collider.CalculateForces(dev_particles, dev_partialPotentialEnergy, N, Lx, Ly, Lz, totalNumCells, dev_cells);

        // Update particle velocities (half-step)
        updateVelocitiesKernel<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5);
        //applyLangevinThermostat<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5, Gamma, T_desired, devStates);


    }


    cout << "-----------------------------------------------------" << endl;




    cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++){
        outFile_positions << i << " " << time << " " << particles[i].GetX() << " " << particles[i].GetY() << " " << particles[i].GetZ() << endl;
        outFile_velocities << i << " " << time << " " << particles[i].GetVelocityX() << " " << particles[i].GetVelocityY() << " " << particles[i].GetVelocityZ() << endl;
        outFile_forces << i << " " << time << " " << particles[i].GetForceX() << " " << particles[i].GetForceY() << " " << particles[i].GetForceZ() << endl;
    }

/*    cudaMemcpy(cells, dev_cells, totalNumCells * sizeof(Cell), cudaMemcpyDeviceToHost);
    cout << " Cell dim : " << numCellsX << " " << numCellsY << " " << numCellsZ << " "<< endl;
    cout << " Number of particles per cell : " << cells[0].numParticles << endl;
    for(int mm = 0; mm < totalNumCells; mm++){
        cout << mm << " ----------------------------" << endl;
        for(int nn = 0; nn < MAX_PARTICLES_PER_CELL; nn++){
            cout << cells[mm].particleIndices[nn] << " ";
        }
        cout << endl;
    }*/

     cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    for(int mm = 0; mm < 10; mm++){
        cout << " Number of particles per particle : " << particles[mm].numNeighbors << endl;
        cout << mm << " ----------------------------" << endl;
        for(int nn = 0; nn < particles[mm].numNeighbors; nn++){
            cout << particles[mm].neighbors[nn] << " ";
        }
        cout << endl;
    }

    cudaMemcpy(cells, dev_cells, totalNumCells * sizeof(Cell), cudaMemcpyDeviceToHost);
    cout << " Cell dim : " << numCellsX << " " << numCellsY << " " << numCellsZ << " "<< endl;
    int mm = 0;
    for(int nn = 0; nn < cells[mm].stencilSize; nn++){
        cout << cells[mm].halfstencil[nn].x << " " << cells[mm].halfstencil[nn].y << " "<< cells[mm].halfstencil[nn].z << endl;
    }

/*    cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    cout << " Number of neighbors : " << particles[0].numNeighbors << endl;
    for(int mm = 0; mm < N; mm++){
        cout << mm << "----------------------------" << endl;
        for(int nn = 0; nn < MAX_NEIGHBORS; nn++){
            cout << particles[mm].neighbors[nn] << " ";
        }
        cout << endl;
    }*/

    std::vector<float> vacf(maxVACFCount, 0.0);
    int vacfCount = 0;
    int vacfSamplingCount = 0;

    std::vector<float> msd(maxMSDCount, 0.0);
    int msdCount = 0;
    int msdSamplingCount = 0;
    for (currentTimeStep = time = drawTime = 0; currentTimeStep < NumberOfSteps; time += dt, drawTime++, currentTimeStep++) {

/*
        if (1){     
            cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);

            for (i = 0; i < N; i++){
                outFile_positions << i << " " << time << " " << particles[i].GetX() << " " << particles[i].GetY() << " " << particles[i].GetZ() << endl;
                outFile_velocities << i << " " << time << " " << particles[i].GetVelocityX() << " " << particles[i].GetVelocityY() << " " << particles[i].GetVelocityZ() << endl;
                outFile_forces << i << " " << time << " " << particles[i].GetForceX() << " " << particles[i].GetForceY() << " " << particles[i].GetForceZ() << endl;
            }
        }*/


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

            float* partialPotentialEnergy = new float[N];
            cudaMemcpy(partialPotentialEnergy, dev_partialPotentialEnergy, N * sizeof(float), cudaMemcpyDeviceToHost);
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
            //cudaMemcpy(stressTensor, dev_stressTensor, 9 * sizeof(float), cudaMemcpyDeviceToHost);
            //outFile_stress << time << " " << stressTensor[0] << " " << stressTensor[1] << " " << stressTensor[2] << " " << stressTensor[3] << " " << stressTensor[4] << " " << stressTensor[5] << " " << stressTensor[6] << " " << stressTensor[7] << " " << stressTensor[8] << endl;

        }  */

        // Update particle velocities (half-step)
        updateVelocitiesKernel<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5);

        // Move particles and calculate displacements
        moveParticlesKernel<<<numBlocks, blockSize>>>(dev_particles, N, dt, Lx, Ly, Lz, dev_maxDisplacement);
        cudaMemcpy(&maxDisplacement, dev_maxDisplacement, sizeof(float), cudaMemcpyDeviceToHost);

        resetCells<<<numBlocks, blockSize>>>(dev_cells, totalNumCells);
        cudaDeviceSynchronize();

        // Assign particles to cells
        assignParticlesToCells<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, cellSize, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
        //identifyNearestNeighborCells<<<numBlocks, blockSize>>>(dev_particles, N, cellSize, numCellsX, numCellsY, numCellsZ, extendedCutoff);
        //cout << maxDisplacement << endl;

        // Update the Verlet list if necessary
        if (maxDisplacement > displacementThreshold) {
            //cout << "-----------------------------------------------------" << endl;
            //updateVerletListKernel<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, extendedCutoff, cellSize, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
            updateVerletListKernel<<<numBlocks, blockSize>>>(dev_particles, dev_cells, N, extendedCutoff, numCellsX, numCellsY, numCellsZ, Lx, Ly, Lz);
            cudaMemset(dev_maxDisplacement, 0, sizeof(float)); // Reset maxDisplacement on the device
            //maxDisplacement = 0 ;
        }

        // Calculate forces
        collider.CalculateForces(dev_particles, dev_partialPotentialEnergy, N, Lx, Ly, Lz, totalNumCells, dev_cells);

        // Update particle velocities (half-step)
        updateVelocitiesKernel<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5);
        //applyLangevinThermostat<<<numBlocks, blockSize>>>(dev_particles, N, dt * 0.5, Gamma, T_desired, devStates);

    }



    //cudaMemcpy(particles, dev_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(dev_particles);

    return 0;
}