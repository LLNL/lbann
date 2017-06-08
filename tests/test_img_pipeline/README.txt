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

In case that we do not know the size of the image (width*height*numChannels), run it as
build/patchWorks image_file

If we know the size of the image, run it as
build/imgpipe image_file size
