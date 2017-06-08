check Elemental_DIR in CMakeList.txt
This is a stand alone installation of Elemental

In addition, check the opencv path, which is a stand alone installation (e.g., on cab or surface)

Make sure if the compiler supports c++11, and the environment viriables, CC and CXX, are set.
e.g.,
CC=gcc
CXX=g++

Then, use the sequence of following commands:
 mkdir build
 cd build
 cmake ..
 make
 cd ..
 export OPENCV_OPENCL_DEVICE=:CPU
 build/patchWorks imgfile 96 48 7 1
 
 The options of the executable are for 96x96 patch size, 48 gap size, 7 jitter size, and
 the centering mode to generate all 8 neighbors around the center patch.
