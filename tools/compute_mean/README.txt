**************************************
        usage of compute_mean
**************************************
Crop images and use them to compute mean. At the same time, store the cropped images.
Computing mean and storing the cropped images are optional. This relies on MPI.


**************************************
        usage of uniform_mean
**************************************
This tool can be used for two tasks. One is to create an image filled with the same pixel
value when the mean pixel value is given per channel as well as the size of the image.
The other is to compute the mean pixel value (for each channel) from an existing image,
and create such an image filled with the mean value.


**************************************
              Building
**************************************
mkdir build; cd build; cmake ..; make

OpenCV and MPI are required to build this tool. The former dependency can be
fullfilled by a system installed copy of OpenCV, or the one built by LBANN.
The variable 'OpenCV_DIR' in CMakeList.txt needs to point to the OpenCV
installation directory.
To use the one built by LBANN, the variable 'LBANN_BUILD_DIR' needs to point to
the LBANN build directory.
Finally, make sure to set the environment variables CC and CXX.


**************************************
             How to run
**************************************
Usage: > ./compute_mean path_file w h [ r rw rh ] bsz save

  path_file: contains the paths to the root data directory, the image list file
             and the output directory. The list file contains the path of each
             image relative to the root directory.
  The parameters w, h, c, rw and rh are for cropper.
    w: the final crop width of image
    h: the final crop height of image.
       (w and h are dictated whether by cropping images to the size)
    r: whether to randomize the crop position within the center region. (0|1)
   rw: The width of the center region with respect to w after resizig the raw image.
   rh: The height of the center region with respect to h after resizing the raw image.
       Raw image will be resized to an image of size rw x rh around the center,
       which covers area of the original image as much as possible while preseving
       the aspect ratio of object in the image.
       When r rw and rh are omitted, it is assumed that r=0, rw=w, and rh=h.

  bsz: The batch size for mean extractor.
       if 0, turns off the mean extractor.
  save: write cropped images. (0|1)


For example, './compute_mean paths.txt 256 256 8192 0'
will take the area around the center of each image as large as possible and resize it to 256x256
without distorting the aspect ratio. Then it will compute the mean of such areas over the given
set of images. The last parameter 0 tells not to save the resized images. The second to the last
parameter 8192 is for batching up the moving average computation. Within a batch, pixel values
accumulates as integer values. Batched sums are converted to floating point values between 0 and
1 for moving average calculation. A larger value poses less risk of precision loss but may lead
to overflow. If this is 0, mean is not computed. 

The first parameter path_file, for instance, contains three paths as below. The first line shows
the root data directory. The second line is a comment, which will be ignored. The third line has
the name of a file in which each line contains a pair of the path of an image file relative to
the root data directory and the label of the image data. The fourth line contains the output path.
--------------------------------------------------------------------
/p/lscratchh/brainusr/datasets/ILSVRC2012/original/train/
#/p/lscratchh/brainusr/datasets/ILSVRC2012/original/labels/train.txt
shortlist.txt
out
--------------------------------------------------------------------


Usage: > ./uniform_mean width height B G R [depth] (for a color image)
    or > ./uniform_mean width height M [depth] (for a monochrome image)
    or > ./uniform_mean input_image [depth]

  B, G, R, M: The mean channel values for blue, green and red for a color image
              or that for the only channel of a monochrome image
       depth: OpenCV depth code for the output image.
              Use 0 for unsigned 8-bit channels, and 2 for unsigned 16-bit channels.
              The default is 0.

  e.g.,  > ./uniform_mean 256 256 104 117 123 0
