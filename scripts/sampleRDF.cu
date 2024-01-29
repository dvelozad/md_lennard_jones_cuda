#include <fstream>
#include <vector>
#include "Particle.h"

__global__ void computeDistancesCUDA(float* x, float* y, float* z, float* distances, int N, float Lx, float Ly, float Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N && i != j) {
        float dx = x[i] - x[j];
        float dy = y[i] - y[j];
        float dz = z[i] - z[j];

        dx -= Lx * round(dx / Lx);
        dy -= Ly * round(dy / Ly);
        dz -= Lz * round(dz / Lz);

        float distance = sqrt(dx * dx + dy * dy + dz * dz);
        distances[i * N + j] = distance;
    }
}

void computeRDFCUDA(Particle* particles, int N, float Lx, float Ly, float Lz, float maxDistance, int numBins, float time, std::ofstream& outFile) {
    // Allocate memory on GPU
    //cout << maxDistance << endl;
    float *dev_x, *dev_y, *dev_z, *dev_distances;
    cudaMalloc(&dev_x, N * sizeof(float));
    cudaMalloc(&dev_y, N * sizeof(float));
    cudaMalloc(&dev_z, N * sizeof(float));
    cudaMalloc(&dev_distances, N * N * sizeof(float));

    // Copy data to GPU
    std::vector<float> x(N), y(N), z(N);
    for (int i = 0; i < N; ++i) {
        x[i] = particles[i].GetX();
        y[i] = particles[i].GetY();
        z[i] = particles[i].GetZ();
    }
    cudaMemcpy(dev_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, z.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(32, 32); 
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch CUDA kernel
    computeDistancesCUDA<<<gridSize, blockSize>>>(dev_x, dev_y, dev_z, dev_distances, N, Lx, Ly, Lz);

    // Copy distances back to CPU
    std::vector<float> distances(N * N);
    cudaMemcpy(distances.data(), dev_distances, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    cudaFree(dev_distances);

    // Histogram the distances and compute RDF
    std::vector<int> bins(numBins, 0);
    float binSize = maxDistance / numBins;
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
    float density = N / (Lx * Ly * Lz);
    std::vector<float> rdf(numBins);
    //std::ofstream outFile(outputFilename);
    for (int i = 0; i < numBins; ++i) {
        float r1 = i * binSize;
        float r2 = r1 + binSize;
        float shellVolume = (4.0 / 3.0) * M_PI * (r2 * r2 * r2 - r1 * r1 * r1);
        rdf[i] = bins[i] / (shellVolume * density * N);
        outFile << time << "\t" << (r1 + r2) / 2 << "\t" << rdf[i] << std::endl;
    }
    //outFile.close();
}