#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <string>
#include "system.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// read in system information
void Readdat(void)
{ 
  FILE *FilePtr;

  char labelBuffer[100];

  FilePtr=fopen("input","r");
  fscanf(FilePtr,"%lf %d %d %lf %lf %lf %d %lf %lf %lf %s\n",   
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

  // print information to the screen
  printf("Molecular Dynamics Program\n");
  printf("\n");
  printf("Number of particles   : %d\n",N);
  printf("Boxlength             : %f\n",Lx);
  printf("Density               : %f\n",N/(Lx*Ly*Lz));
  printf("Temperature           : %f\n",T_desired);
  printf("Cut-Off radius        : %f\n",cutoff);
  printf("Cut-Off energy        : %f\n",Ecut);
  printf("Number of uteps       : %d\n",NumberOfSteps);
  printf("Number of init steps  : %d\n",equilibrationSteps);
  printf("Timestep              : %f\n",dt);
}
