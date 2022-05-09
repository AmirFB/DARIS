clear
rm test
nvcc -lcuda -lineinfo -I src src/ctx.cu src/schd.cpp test.cpp -o test
./test