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
float Gamma=0.1;
float T_desired=2;

// Particles definitions
float defaultMass=1;
float InitialVelocity=20;

// Simul box settings
float RHO;
int M;
int N;
float L =1;
float Lx=1;
float Ly=1;
float Lz=1;

