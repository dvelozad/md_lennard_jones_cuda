#include <cmath>
#include <string>

// Simulation label
extern std::string simulationLabel;

// time step
extern double dt;
extern int equilibrationSteps;
//const double totalTime = 10; //In reduced units
extern int NumberOfSteps;

// Lennard-Jones
extern double kB;
extern double epsilon;
extern double sigma;
extern double cutoff;
extern double Ecut;
extern double buffer;

// Thermostat
extern double Gamma;
extern double T_desired;

// Particles definitions
extern double defaultMass;
extern double InitialVelocity;

// Simul box settings
extern double RHO;
extern int M;
extern int N;
extern double L;
extern double Lx;
extern double Ly;
extern double Lz;

void Readdat(void);