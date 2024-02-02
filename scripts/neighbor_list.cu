#include "Particle.h"
#include "system.h"


__global__ void resetCells(Cell* cells, int totalNumCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalNumCells) {
        cells[idx].numParticles = 0;
        cells[idx].overflow = false; 

        for (int nPart = 0; nPart < MAX_PARTICLES_PER_CELL; nPart++){
            cells[idx].particleIndices[nPart] = 0;
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

__global__ void assignParticlesToCells(Particle* particles, Cell* cells, int N, float cellSize, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) { // Use >= to ensure we don't access out-of-bounds memory if i == N
        return;
    }

    Particle* pi = &particles[i];

    // Assume particle positions are always positive; otherwise adjust as needed
    float wrappedX = fmodf(pi->x, Lx);
    float wrappedY = fmodf(pi->y, Ly);
    float wrappedZ = fmodf(pi->z, Lz);

    // Calculate cell index for particle i
    int cellX = static_cast<int>(wrappedX / cellSize);
    int cellY = static_cast<int>(wrappedY / cellSize);
    int cellZ = static_cast<int>(wrappedZ / cellSize);

    // Ensure that the cell indices are within the domain
    cellX = (cellX + numCellsX) % numCellsX;
    cellY = (cellY + numCellsY) % numCellsY;
    cellZ = (cellZ + numCellsZ) % numCellsZ;

    // Linearize the 3D cell index to a 1D index
    int cellIndex = cellZ * numCellsY * numCellsX + cellY * numCellsX + cellX;

    // Check for valid cellIndex and handle the case where cellIndex is out-of-bounds
    if (cellIndex < numCellsX * numCellsY * numCellsZ) {
        int idx = atomicAdd(&cells[cellIndex].numParticles, 1);
        
        // Use a separate if-statement to handle the idx value after the atomic operation
        if (idx < MAX_PARTICLES_PER_CELL) {
            // Additional check might be needed here to ensure idx is still within bounds
            cells[cellIndex].particleIndices[idx] = i;
        } 
    }
    // Store the cell index in the Particle structure
    pi->cellIndex = cellIndex;
}


__global__ void updateVerletListKernel(Particle* particles, Cell* cells, int N, float cutoff_, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Select particle i
        Particle* pi = &particles[i];

        // Reset num of neighbors
        pi->numNeighbors = 0;

        // Reset last pos
        pi->lastX = pi->x;
        pi->lastY = pi->y;
        pi->lastZ = pi->z;  
        
        // Get the index of the cell where particle i is located
        int cellIndex = pi->cellIndex;

        // Get the cell structure for particle i's cell
        Cell* cell = &cells[cellIndex];

        // Loop over the stencil of the cell for neighboring cell offsets
        for (int n = 0; n < cell -> stencilSize; n++) {
        //for (int n = 0; n < 1; n++) {
            int3 offset = cell -> halfstencil[n];
            int neighborCellIndex = getNeighborCellIndex(cellIndex, offset.x, offset.y, offset.z, numCellsX, numCellsY, numCellsZ);

            // Loop over particles in neighboring cell
            for (int j = 0; j < cells[neighborCellIndex].numParticles; j++) {
            //    for (int j = 0; j < MAX_PARTICLES_PER_CELL; j++) {
                int neighborIndex = cells[neighborCellIndex].particleIndices[j];
                // Avoid double counting: only consider particles in neighboring cells
                //if (neighborIndex != i) {
                if (i > neighborIndex) {
            
                    float dx_ = pi->x - particles[neighborIndex].x;
                    float dy_ = pi->y - particles[neighborIndex].y;
                    float dz_ = pi->z - particles[neighborIndex].z;

                    // Apply minimum image convention
                    dx_ -= Lx * round(dx_ / Lx);
                    dy_ -= Ly * round(dy_ / Ly);
                    dz_ -= Lz * round(dz_ / Lz);

                    float rsq = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                    if (rsq < cutoff_ * cutoff_) {
                        if (pi->numNeighbors < MAX_NEIGHBORS) {
                            pi->neighbors[pi->numNeighbors++] = neighborIndex;
                        }
                    }
                }
            }
        }
    }
}


__global__ void generateStencils(Cell* cells, int numCellsX, int numCellsY, int numCellsZ, float cellLength, float cutoff) {
    int cellIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int totalCells = numCellsX * numCellsY * numCellsZ;

    if (cellIndex >= totalCells) return;

    Cell* cell = &cells[cellIndex];
    cell->stencilSize = 0; // Initialize stencil size

    // Calculate the maximal extent of the stencil based on the cutoff
    int maxDx = ceil(cutoff / cellLength);
    int maxDy = ceil(cutoff / cellLength);
    int maxDz = ceil(cutoff / cellLength);

    // Generate the stencil considering the maximal extent
    for (int dx = -maxDx; dx <= maxDx; dx++) {
        for (int dy = -maxDy; dy <= maxDy; dy++) {
            for (int dz = -maxDz; dz <= maxDz; dz++) {
                // Skip the center cell to only consider the surrounding cells
                //if (dx == 0 && dy == 0 && dz == 0) continue;

                    // If within the cutoff, add the cell offset to the stencil
                    if (cell->stencilSize < MAX_STENCIL_SIZE) {
                        cell->stencil[cell->stencilSize++] = make_int3(dx, dy, dz);
                    }
                }
            }
        }

    // Now that we have a full stencil, let's reduce it to a half stencil
    // by only including neighbor cells in the positive direction.
    int halfStencilCount = 0;
    for (int i = 0; i < cell->stencilSize; ++i) {
        int3 offset = cell -> stencil[i];
        if (offset.x >= 0 && offset.y >= 0 && offset.z >= 0) {
            cell->halfstencil[halfStencilCount++] = offset;
        }
    }
    cell->stencilSize = halfStencilCount; // Update the stencil size to the count of half stencil
}


/*__global__ void updateVerletListKernel(Particle* particles, Cell* cells, int N, float cutoff_, float cellSize, int numCellsX, int numCellsY, int numCellsZ, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Particle* pi = &particles[i];
        pi->numNeighbors = 0;

        int cellIndex = pi->cellIndex;

            // Loop over neighboring cells including the current cell
        for (int dx = 0; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    if (dx == 0 && dy == 0 && dz == -1) continue; // Skip the current cell

                    int neighborCellIndex = getNeighborCellIndex(cellIndex, dx, dy, dz, numCellsX, numCellsY, numCellsZ);

                    // Loop over particles in neighboring cell
                    for (int j = 0; j < cells[neighborCellIndex].numParticles; j++) {
                        int neighborIndex = cells[neighborCellIndex].particleIndices[j];
                        if (neighborIndex != i) { // Avoid double counting
                            float dx_ = pi->x - particles[neighborIndex].x;
                            float dy_ = pi->y - particles[neighborIndex].y;
                            float dz_ = pi->z - particles[neighborIndex].z;

                            // Apply minimum image convention
                            dx_ -= Lx * round(dx_ / Lx);
                            dy_ -= Ly * round(dy_ / Ly);
                            dz_ -= Lz * round(dz_ / Lz);

                            float rsq = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                            if (rsq < cutoff_ * cutoff_) {
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
*/