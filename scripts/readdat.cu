#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string>

#include "system.h"
#include "cuda_opt_constants.h"

#include "write_settings.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// read in system information
void Readdat(void)
{ 
  FILE *FilePtr;

  char labelBuffer[100];

  FilePtr=fopen("input","r");
  //fscanf(FilePtr,"%lf %d %d %lf %lf %lf %d %lf %lf %lf %s\n",   
  fscanf(FilePtr, "%f %d %d %f %f %f %d %f %f %f %s\n", 
         &RHO,
         &M,
         &NumberOfSteps,
         &T_desired,
         &Gamma,
         &dt,
         &equilibrationSteps,
         &defaultMass,
         &InitialVelocity,
         &cutoff,
         labelBuffer);

  simulationLabel = labelBuffer;

  fclose(FilePtr);

  // Calculate Some Parameters
  Ecut=4.0*(pow((cutoff * cutoff),-6.0)-pow((cutoff * cutoff),-3.0));

  // Number of particles
  N = 4*M*M*M;

  // Box dim
  L = pow(defaultMass*N / RHO, 1. / 3.);
  Lx = L, Ly = L, Lz = L;

  //simulationLabel = "T" + std::to_string((int)T_desired) + "_N" + std::to_string((int)M)+ "_RHO" + std::to_string((int)RHO);

  sigma_6 = pow(sigma, 6);
  epsilon_sigma_6 = epsilon * sigma_6;

  potentialEnergy_cutoff = 4 * epsilon_sigma_6 * pow(1 / cutoff, 6) * (sigma_6 * pow(1 / cutoff, 6) - 1);
  forceNormalCutOff = 24 * epsilon_sigma_6 * pow(1 / cutoff, 6) * pow(1 / cutoff, 2) * ( 2 * sigma_6 * pow(1 / cutoff, 6) - 1);

  FilePtr=fopen("write_input","r");
  fscanf(FilePtr, "%d %d %d %d %d %d %d %d %d %f %d\n", 
         &RDF_writeFrame,
         &temperature_writeFrame,
         &eqVerboseFrame,
         &vacf_writeFrame,
         &maxVACFCount,
         &vacfSamplingReps,
         &msd_writeFrame,
         &maxMSDCount,
         &msdSamplingReps,
         &maxDistance,
         &numBins);

  // print information to the screen
  printf("Molecular Dynamics Program\n");
  printf("\n");
  printf("Simulation label      : %s\n",labelBuffer);
  printf("Number of particles   : %d\n",N);
  printf("Boxlength             : %f\n",Lx);
  printf("Density               : %f\n",N/(Lx*Ly*Lz));
  printf("Temperature           : %f\n",T_desired);
  printf("Gamma                 : %f\n",Gamma);
  printf("Cut-Off radius        : %f\n",cutoff);
  printf("Cut-Off energy        : %f\n",Ecut);
  printf("Number of uteps       : %d\n",NumberOfSteps);
  printf("Number of init steps  : %d\n",equilibrationSteps);
  printf("Timestep              : %f\n",dt);
}
