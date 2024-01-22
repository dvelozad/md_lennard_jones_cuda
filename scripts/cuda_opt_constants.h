#include <cmath>
#include "system.h"

__device__ double kB_d;

// Performance definitions
const double sigma_6 = pow(sigma, 6);
const double epsilon_sigma_6 = epsilon * sigma_6;

const double potentialEnergy_cutoff = 4 * epsilon_sigma_6 * pow(1 / cutoff, 6) * (sigma_6 * pow(1 / cutoff, 6) - 1);
const double forceNormalCutOff = 24 * epsilon_sigma_6 * pow(1 / cutoff, 6) * pow(1 / cutoff, 2) * ( 2 * sigma_6 * pow(1 / cutoff, 6) - 1);
