#include "Particle.h"

#include <string>
#include <fstream>

// Write definitions
extern int RDF_writeFrame;

extern int temperature_writeFrame;

extern int eqVerboseFrame;

extern int vacf_writeFrame;
extern int maxVACFCount;
extern int vacfSamplingReps;

extern int msd_writeFrame;
extern int maxMSDCount;
extern int msdSamplingReps;

// Radial distribution function definitions
extern float maxDistance;
extern int numBins;

float CalculateCurrentTemperature(Particle *particles, int numParticles);
void ComputeMSD(Particle* particles, float* msd, int N, float time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite);
void ComputeVACF(Particle* particles, float* vacf, int N, float time, int count, std::ofstream& outFile, bool shouldReset, bool shouldWrite);
void computeRDFCUDA(Particle* particles, int N, float Lx, float Ly, float Lz, float maxDistance, int numBins, float time, std::ofstream& outFile);

class OutputManager {
public:
    void setOutputNames(const std::string& simulationLabel);
    void openFiles();

    // Accessors for ofstream objects
    std::ofstream& getPosFile();
    std::ofstream& getVelFile();
    std::ofstream& getForcesFile();
    std::ofstream& getRdfFile();
    std::ofstream& getVafFile();
    std::ofstream& getMsdFile();
    std::ofstream& getTemperatureFile();

private:
    std::string rdfFilename;
    std::string vafFilename;
    std::string msdFilename;
    std::string temperatureFilename;

    std::string velocitiesFilename;
    std::string forcesFilename;
    std::string stressFilename;
    std::string energyFilename;
    std::string positionsFilename;

    std::ofstream outFile_positions;
    std::ofstream outFile_velocities;
    std::ofstream outFile_forces;
    std::ofstream outFile_rdf;
    std::ofstream outFile_vacf;
    std::ofstream outFile_msd;
    std::ofstream outFile_temperature;
};
