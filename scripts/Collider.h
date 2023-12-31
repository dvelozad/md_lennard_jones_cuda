// Collider.h
#ifndef COLLIDER_H
#define COLLIDER_H

#include "Particle.h"

__device__ double epsilon_sigma_6_d, sigma_6_d, forceNormalCutOff_d;

class Collider {
private:
    double potentialEnergy;

public:
    void Init(void);
    //void CalculateForces(Particle *particles);
    void CalculateForces(Particle* dev_particles, double* dev_partialPotentialEnergy, int N);
    //void Collide(Particle &particle1, Particle &particle2);
    double GetPotentialEnergy(void);
};

#endif
