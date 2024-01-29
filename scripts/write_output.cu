#include <iostream>
#include <fstream>
#include <string>

#include "system.h"
#include "write_settings.h"

// Write definitions
int RDF_writeFrame;

int temperature_writeFrame;

int eqVerboseFrame;

int vacf_writeFrame;
int maxVACFCount;
int vacfSamplingReps;

int msd_writeFrame;
int maxMSDCount;
int msdSamplingReps;

// Radial distribution function definitions
float maxDistance;
int numBins;

void OutputManager::setOutputNames(const std::string& simulationLabel) {
    velocitiesFilename  = "../output_files/" + simulationLabel + '/' + simulationLabel + "_velocities_data.txt";
    forcesFilename      = "../output_files/" + simulationLabel + '/' + simulationLabel + "_forces_data.txt";
    stressFilename      = "../output_files/" + simulationLabel + '/' + simulationLabel + "_stress_tensor_data.txt";
    energyFilename      = "../output_files/" + simulationLabel + '/' + simulationLabel + "_energy_data.txt";
    positionsFilename   = "../output_files/" + simulationLabel + '/' + simulationLabel + "_positions_data.txt";
    rdfFilename         = "../output_files/" + simulationLabel + '/' + simulationLabel + "_rdf_data.txt";
    vafFilename         = "../output_files/" + simulationLabel + '/' + simulationLabel + "_vaf_data.txt";
    msdFilename         = "../output_files/" + simulationLabel + '/' + simulationLabel + "_msd_data.txt";
    temperatureFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_temperature_data.txt";
}

void OutputManager::openFiles() {
    outFile_positions.open(positionsFilename);
    outFile_velocities.open(velocitiesFilename);
    outFile_forces.open(forcesFilename);
    outFile_rdf.open(rdfFilename);
    outFile_vacf.open(vafFilename);
    outFile_msd.open(msdFilename);
    outFile_temperature.open(temperatureFilename);
}

std::ofstream& OutputManager::getPosFile() {
    return outFile_positions;
}

std::ofstream& OutputManager::getVelFile() {
    return outFile_velocities;
}

std::ofstream& OutputManager::getForcesFile() {
    return outFile_forces;
}

std::ofstream& OutputManager::getRdfFile() {
    return outFile_rdf;
}

std::ofstream& OutputManager::getVafFile() {
    return outFile_vacf;
}

std::ofstream& OutputManager::getMsdFile() {
    return outFile_msd;
}

std::ofstream& OutputManager::getTemperatureFile() {
    return outFile_temperature;
}



/*   
// To save energies
energyFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_energy_data.txt";
std::ofstream outFile_energy(energyFilename);
if (!outFile_energy) {
    cerr << "Error opening file for writing" << endl;
    return 1;
}

// To save positions
positionsFilename = "../output_files/" + simulationLabel + '/' + simulationLabel + "_positions_data.txt";
std::ofstream outFile_positions(positionsFilename);
if (!outFile_positions) {
    cerr << "Error opening file for writing" << endl;
    return 1; 
}
*/

/*    // To save positions
std::ofstream outFile_velocities(velocitiesFilename);
if (!outFile_velocities) {
    cerr << "Error opening file for writing" << endl;
    return 1; 
}

// To save positions
std::ofstream outFile_forces(forcesFilename);
if (!outFile_forces) {
    cerr << "Error opening file for writing" << endl;
    return 1; 
}

// To save positions
std::ofstream outFile_stress(stressFilename);
if (!outFile_stress) {
    cerr << "Error opening file for writing" << endl;
    return 1; 
}*/
