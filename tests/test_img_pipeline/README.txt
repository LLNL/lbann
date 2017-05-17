check Elemental_DIR in CMakeList.txt

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
