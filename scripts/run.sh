#!/bin/bash

temperature=$1
N_=$2
RHO_=$3

# label
simulationLabel="T${temperature}_N${N_}_RHO${RHO_}"

# Define the directory path
outputDir="../output_files/${simulationLabel}"

# Create the directory if it doesn't exist
mkdir -p "${outputDir}"

# Compile CUDA source files with nvcc
nvcc -c Particle.cu
nvcc -c Collider.cu
nvcc -c main.cu

# Check for compilation errors
if [ $? -ne 0 ]; then
    echo "Compilation error."
    exit 1
fi

# Link the object files and create the executable
nvcc Particle.o Collider.o main.o -o simul_exec-$temperature-$N_-$RHO_

# Check for linking errors
if [ $? -ne 0 ]; then
    echo "Linking error."
    exit 1
fi

# Run the executable
./simul_exec-$temperature-$N_-$RHO_
