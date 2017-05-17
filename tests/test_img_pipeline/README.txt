mkdir build
cd build
cmake ..
make
cd ..
export OPENCV_OPENCL_DEVICE=:CPU
build/patchWorks img/chalk-color-red-teacher-158717.jpeg 96 48 7 1
