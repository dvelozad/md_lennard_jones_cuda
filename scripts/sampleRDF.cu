#include <fstream>
#include <vector>
#include "Particle.h"

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
        outFile << time << "\t" << (r1 + r2) / 2 << "\t" << rdf[i] << std::endl;
    }
    //outFile.close();
}