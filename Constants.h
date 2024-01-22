// Constants.h

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef T
#define T 0.71
#endif

#ifndef N_
#define N_ 20
#endif

#ifndef RHO_
#define RHO_ 1
#endif

// Simulation label
#include <cmath>
#include <string>
const std::string simulationLabel = "T" + std::string(TOSTRING(T)) + "_N" + std::string(TOSTRING(N_))+ "_RHO" + std::string(TOSTRING(RHO_));


// Lennard-Jones
const double kB         = 1;
const double epsilon    = 1;
const double sigma      = 1;
const double cutoff     = 2.5;
const double buffer     = 2;

// For Omelyan PEFRL
const double Zeta   = 0.1786178958448091;
const double Lambda = -0.2123418310626054;
const double Xi     = -0.06626458266981849;

const double Gamma      = 3;
const double T_desired  = T;
//const double Q          = 1; 

// Particles definitions
const double defaultMass = 1;
const double InitialVelocity = 21;
//const int Nx = NX, Ny = NY, Nz = NZ, N = Nx * Ny * Nz;
const int N = 4*N_*N_*N_;


const double RHO = RHO_;
const double L = pow(defaultMass*N / RHO, 1. / 3.);
const double Lx = L, Ly = L, Lz = L;

//extern __device__ double Lx_d, Ly_d, Lz_d;

// Write definitions
const double dt = 0.005;

const int RDF_writeFrame = 100;
const int temperature_writeFrame = 500;

const double equilibrationSteps = 1001;
const double totalTime = 10; //In reduced units

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

// Performance definitions
const double sigma_6 = pow(sigma, 6);
const double epsilon_sigma_6 = epsilon * sigma_6;

const double potentialEnergy_cutoff = 4 * epsilon_sigma_6 * pow(1 / cutoff, 6) * (sigma_6 * pow(1 / cutoff, 6) - 1);
const double forceNormalCutOff = 24 * epsilon_sigma_6 * pow(1 / cutoff, 6) * pow(1 / cutoff, 2) * ( 2 * sigma_6 * pow(1 / cutoff, 6) - 1);

#endif