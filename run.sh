#!/bin/sh
cd /Users/jerome/Documents/EPFL/1ere/Progra\ 2/cpp/Neural-Network
rm build
mkdir build
cd build
cmake ..
make
./projet_network
cd ../src
./graphs.py
cd ../build