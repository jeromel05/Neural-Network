#!/bin/sh
cd /Users/jerome/Documents/Prog\ Gen/cpp/NeuralNetNew 
mkdir build
cd build
cmake ..
make
./projet_network
cd ../src
./graphs.py
cd ../build