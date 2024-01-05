#!/bin/bash

T=0.71
N=6

RHO_values=(0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5) 

constants_file="./Constants.h"

for RHO in "${RHO_values[@]}"; do
    sed -i "s/#define RHO_ .*/#define RHO_ $RHO/" "$constants_file"
    ./run.sh "$T" "$N" "$RHO"
    wait
done
