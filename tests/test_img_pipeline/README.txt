check Elemental_DIR in CMakeList.txt
This requires OpenCV, Elemental, and MPI. cmake will attempt to find these
under system or LBANN build directories.
To set the LBANN build directory, set CLUSTER variable in CMakeList.txt

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

run it as
build/imgpipe image_filename w h r rw rh bsz a n ni

    The parameters w, h, c, rw and rh are for cropper
    w: the final crop width of image
    h: the final crop height of image
       (w and h are dictated whether by cropping images to the size)
    r: whether to randomize the crop position within the center region (0|1)
   rw: The width of the center region with respect to w after resizig the raw image
   rh: The height of the center region with respect to h after resizing the raw image
       Raw image will be resized to an image of size rw x rh around the center,
       which covers area of the original image as much as possible while preseving
       the aspect ratio of object in the image

  bsz: The batch size for mean extractor
       if 0, turns off te mean extractor

    a: whether to use augmenter (0|1)

    n: whether to use normalizer (0|1)

   ni: The number of iterations.
       must be greater than 0

e.g., build/imgpipe img.jpg 240 240 1 256 256 4 0 0 8
