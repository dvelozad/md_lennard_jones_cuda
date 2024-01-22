#include "Particle.h"

// Write definitions
const int RDF_writeFrame = 100;
const int temperature_writeFrame = 500;

const int eqVerboseFrame = 100;

const int vacf_writeFrame = 1;
const int maxVACFCount = 10;
const int vacfSamplingReps = 10;

const int msd_writeFrame = 1;
const int maxMSDCount = 10;
const int msdSamplingReps = 10;

// Radial distribution function definitions
const double maxDistance = 10; 
const int numBins = 1000; 


double CalculateCurrentTemperature(Particle *particles, int numParticles);
void ComputeMSD(Particle* particles, double* msd, int N, double time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite);
void ComputeVACF(Particle* particles, double* vacf, int N, double time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite);
void computeRDFCUDA(Particle* particles, int N, double Lx, double Ly, double Lz, double maxDistance, int numBins, double time, std::ofstream& outFile);