// Particle.cpp
#include "Particle.h"
#include "Constants.h"
#include <cmath>
#include <random>
#include <iostream>

using namespace std;

double Particle::GetMass(void) { return mass; }
double Particle::GetX(void) { return x; }
double Particle::GetY(void) { return y; }
double Particle::GetZ(void) { return z; }
double Particle::GetKineticEnergy(void) { return 0.5*mass * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ); }
double Particle::GetVelocity(void) { return sqrt(velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ); }
double Particle::GetVelocityX(void) { return velocityX; }
double Particle::GetVelocityY(void) { return velocityY; }
double Particle::GetVelocityZ(void) { return velocityZ; }