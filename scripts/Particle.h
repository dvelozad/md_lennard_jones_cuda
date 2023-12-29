#ifndef PARTICLE_H
#define PARTICLE_H

#include "Constants.h"

__device__ double Lx_d, Ly_d, Lz_d;

class Particle {
public:
    double mass, radius;
    double x, y, z; // Position
    double velocityX, velocityY, velocityZ; // Velocity
    double forceX, forceY, forceZ; // Force
    double xi = 0.0; // Thermostat variable
    double xidot = 0.0; // Time derivative of the thermostat variable
    double lastX, lastY, lastZ; // Store the positions at last neighbor list update

    __host__ __device__ void Init(double x0, double y0, double z0, double velocityX0, double velocityY0, double velocityZ0, double mass0, double radius0) {
        x = x0; y = y0; z = z0;
        velocityX = velocityX0; velocityY = velocityY0; velocityZ = velocityZ0;
        mass = mass0; radius = radius0;
    }
    double GetMass(void);
    double GetX(void);
    double GetY(void);
    double GetZ(void);
    double GetKineticEnergy(void);
    double GetVelocity(void);
    double GetVelocityX(void);
    double GetVelocityY(void);
    double GetVelocityZ(void);
    double GetNeighborList(void);

    __host__ __device__ void ApplyPeriodicBoundaryConditions(double Lx, double Ly, double Lz) {
        if (x < 0) x += Lx;
        else if (x > Lx) x -= Lx;

        if (y < 0) y += Ly;
        else if (y > Ly) y -= Ly;

        if (z < 0) z += Lz;
        else if (z > Lz) z -= Lz;
    }

    __device__ void Move_r(double dt) {
        x += velocityX * dt;
        y += velocityY * dt;
        z += velocityZ * dt;
        ApplyPeriodicBoundaryConditions(Lx_d, Ly_d, Lz_d);
    }

    __device__ void Move_V(double dt) {
        velocityX += (forceX / mass) * dt;
        velocityY += (forceY / mass) * dt;
        velocityZ += (forceZ / mass) * dt;
    }
};

#endif
