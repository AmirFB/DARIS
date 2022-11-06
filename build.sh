clear
mkdir build
cd build
rm fgprs
cmake -DCMAKE_PREFIX_PATH=/home/amir/repos/pytorch-install2/ ..
cmake --build . --config Release