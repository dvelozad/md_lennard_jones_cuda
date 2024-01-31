#include "Particle.h"
#include "system.h"

/*__global__ void updateVerletListKernel(Particle* particles, int N, float extendedCutoff, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Particle* pi = &particles[i];
        pi -> numNeighbors = 0;

       // Store current position as last position
        pi->lastX = pi->x;
        pi->lastY = pi->y;
        pi->lastZ = pi->z;

        for (int j = 0; j < N; j++) {
            if (i != j) {
                // Calculate the distance between particles[i] and particles[j]
                float dx = pi->x - particles[j].x;
                float dy = pi->y - particles[j].y;
                float dz = pi->z - particles[j].z;

                // Apply minimum image convention
                dx -= Lx * round(dx / Lx);
                dy -= Ly * round(dy / Ly);
                dz -= Lz * round(dz / Lz);

                float rsq = dx * dx + dy * dy + dz * dz;

                // Check if the particle j is within the extended cutoff distance
                if (rsq < extendedCutoff * extendedCutoff) {
                    // Add particle j to the neighbor list of particle i
                    if (pi->numNeighbors < MAX_NEIGHBORS) {
                        pi->neighbors[pi->numNeighbors] = j;
                        pi->numNeighbors++;
                    } 
                }
            }
        }
    }
}*/



__global__ void resetCells(Cell* cells, int totalNumCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalNumCells) {
        cells[idx].numParticles = 0;
        cells[idx].overflow = false; // Assuming you have an overflow flag

        for (int nPart = 0; nPart < MAX_PARTICLES_PER_CELL; nPart++){
            cells[idx].particleIndices[nPart] = 0;
        }
    }
}

__global__ void assignParticlesToCells(Particle* particles, Cell* cells, int N, float cellSize, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Particle* pi = &particles[i];

        // Handle periodic boundary conditions
        float wrappedX = fmodf(pi->x + Lx, Lx);
        float wrappedY = fmodf(pi->y + Ly, Ly);
        float wrappedZ = fmodf(pi->z + Lz, Lz);

        // Calculate cell index for particle i
        int cellX = static_cast<int>(wrappedX / cellSize);
        int cellY = static_cast<int>(wrappedY / cellSize);
        int cellZ = static_cast<int>(wrappedZ / cellSize);

        // Linearize the 3D cell index to a 1D index
        int cellIndex = (cellZ * numCellsY + cellY) * numCellsX + cellX;
        pi->cellIndex = cellIndex; // Store the cell index in the Particle structure

        // Safely add particle i to its cell
        if (cellIndex < numCellsX * numCellsY * numCellsZ) {
            int idx = atomicAdd(&cells[cellIndex].numParticles, 1);
            if (idx < MAX_PARTICLES_PER_CELL) {
                cells[cellIndex].particleIndices[idx] = i;
            } else {
                // If the cell is already full, set the overflow flag
                cells[cellIndex].overflow = true;
                // Optionally, you could use an auxiliary list to store overflow particles
            }
        }
    }
}


__device__ int getNeighborCellIndex(int cellIndex, int dx, int dy, int dz, int numCellsX, int numCellsY, int numCellsZ) {
    // Compute 3D cell indices from linear index
    int cellZ = cellIndex / (numCellsX * numCellsY);
    int cellY = (cellIndex / numCellsX) % numCellsY;
    int cellX = cellIndex % numCellsX;

    // Apply periodic boundary conditions
    cellX = (cellX + dx + numCellsX) % numCellsX;
    cellY = (cellY + dy + numCellsY) % numCellsY;
    cellZ = (cellZ + dz + numCellsZ) % numCellsZ;

    // Linearize and return the new cell index
    return cellZ * numCellsY * numCellsX + cellY * numCellsX + cellX;
}

__global__ void updateVerletListKernel(Particle* particles, Cell* cells, int N, float cutoff_, float cellSize, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Particle* pi = &particles[i];
        pi->numNeighbors = 0;

        int cellIndex = pi->cellIndex; // Assuming cellIndex is already calculated

        pi->lastX = pi->x;
        pi->lastY = pi->y;
        pi->lastZ = pi->z;

        // Loop over neighboring cells
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int neighborCellIndex = getNeighborCellIndex(cellIndex, dx, dy, dz, numCellsX, numCellsY, numCellsZ);

                    // Loop over particles in neighboring cell
                    for (int j = 0; j < cells[neighborCellIndex].numParticles; j++) {
                        int neighborIndex = cells[neighborCellIndex].particleIndices[j];
                        if (neighborIndex != i) {
                            // Compute distance between particles i and neighbor
                            float dx_ = pi->x - particles[neighborIndex].x;
                            float dy_ = pi->y - particles[neighborIndex].y;
                            float dz_ = pi->z - particles[neighborIndex].z;

                            // Apply minimum image convention
                            dx_ -= Lx * round(dx_ / Lx);
                            dy_ -= Ly * round(dy_ / Ly);
                            dz_ -= Lz * round(dz_ / Lz);

                            float rsq = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                            if (rsq < cutoff_ * cutoff_) {
                                //pi->neighbors[pi->numNeighbors] = neighborIndex;
                                //pi->numNeighbors++;

                                if (pi->numNeighbors < MAX_NEIGHBORS) {
                                    pi->neighbors[pi->numNeighbors++] = neighborIndex;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}