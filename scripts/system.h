#include <cmath>
#include <string>

// Simulation label
extern std::string simulationLabel;

// time step
extern float dt;
extern int equilibrationSteps;
//const float totalTime = 10; //In reduced units
extern int NumberOfSteps;

// Lennard-Jones
extern float kB;
extern float epsilon;
extern float sigma;
extern float cutoff;
extern float Ecut;
extern float buffer;

// Thermostat
extern float Gamma;
extern float T_desired;

// Particles definitions
extern float defaultMass;
extern float InitialVelocity;

// Simul box settings
extern float RHO;
extern int M;
extern int N;
extern float L;
extern float Lx;
extern float Ly;
extern float Lz;


extern float maxDisplacement;
extern float extendedCutoff;
extern float displacementProportion;

extern float skin;

void Readdat(void);