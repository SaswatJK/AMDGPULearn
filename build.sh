#!/bin/bash
name=$1
hipcc -O3 -march=native "$name.hip" -o "build/$neuralNet" -lhiprand
