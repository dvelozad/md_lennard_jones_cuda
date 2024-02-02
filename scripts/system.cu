#include "system.h"
#include "cuda_opt_constants.h"

// Simulation label
std::string simulationLabel;

// time step
float dt;
int equilibrationSteps;
int NumberOfSteps;

// Lennard-Jones
float kB       = 1;
float epsilon  = 1;
float sigma    = 1;
float cutoff;
float Ecut;
float buffer;

float sigma_6;
float epsilon_sigma_6;
float potentialEnergy_cutoff;
float forceNormalCutOff;

// Thermostat
float Gamma;
float T_desired;

// Particles definitions
float defaultMass;
float InitialVelocity;

// Simul box settings
float RHO;
int M;
int N;
float L;
float Lx;
float Ly;
float Lz;

float maxDisplacement;
float extendedCutoff;
float displacementProportion;

float skin;