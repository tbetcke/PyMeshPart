#!/usr/bin/sh
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-4.0.tar.gz
tar xzvf ./metis-4.0.tar.gz
cd metis-4.0
patch -p1 < ../metis-4.0.patch
cd Lib
cp *.c ../../src/
cp *.h ../../src/
cd ../../



