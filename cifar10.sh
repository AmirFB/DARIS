clear
rm test
nvcc -lcuda -lineinfo -o cif10 -I src -I src/dat -I src/net cif10.cpp src/net/resnet.cpp src/dat/cif10.cpp