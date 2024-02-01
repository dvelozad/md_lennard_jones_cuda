// Collider.h
#ifndef COLLIDER_H
#define COLLIDER_H

#include "Particle.h"

class Collider {
private:
    float potentialEnergy;

public:
    void Init(void);
    //void CalculateForces(Particle *particles);
    //void CalculateForces(Particle* dev_particles, float* dev_partialPotentialEnergy, int N, float Lx, float Ly, float Lz);
    void CalculateForces(Particle* dev_particles, float* dev_partialPotentialEnergy, int N, float Lx, float Ly, float Lz, int totalCells, Cell* cells);
    //void Collide(Particle &particle1, Particle &particle2);
    float GetPotentialEnergy(void);
};

#endif
