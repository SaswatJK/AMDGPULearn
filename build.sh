#!/bin/bash
name=$1
cd build
hipcc -O3 -march=native "../$name" -o "nn" -lhiprand
