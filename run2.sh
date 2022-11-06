clear
rm test
nvcc -lcuda -lineinfo -o test -I src src/schd.cpp src/ctx.cu test.cpp
# nvcc -lcuda -lineinfo -I src src/ctx.cu test.cpp -o test
./test