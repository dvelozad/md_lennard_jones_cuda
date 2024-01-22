#include "system.h"

// Simulation label
std::string simulationLabel;

// time step
double dt;
int equilibrationSteps;
int NumberOfSteps;

// Lennard-Jones
double kB;
double epsilon;
double sigma;
double cutoff;
double Ecut;
double buffer;

// Thermostat
double Gamma;
double T_desired;

// Particles definitions
double defaultMass;
double InitialVelocity;

// Simul box settings
double RHO;
int M;
int N;
double L;
double Lx;
double Ly;
double Lz;