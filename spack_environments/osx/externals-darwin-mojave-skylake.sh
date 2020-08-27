#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi:
          - openmpi@4.0 arch=darwin-mojave-skylake
        blas:
          - veclibfort
        lapack:
          - veclibfort
      buildable: true
      version: []
EOF
)

EXTERNAL_PACKAGES=$(cat <<EOF
    cmake::
      buildable: True
      variants: ~openssl ~ncurses
      version:
      - 3.16.2
      externals:
      - spec: cmake@3.16.2 arch=darwin-mojave-skylake
        prefix:  /usr/local/

    hwloc::
      buildable: True
      version:
      - 2.0.2

    llvm::
      buildable: False
      variants: +clang
      version:
      - 9.0.0
      externals:
      - spec: llvm@9.0.0 arch=darwin-mojave-skylake
        prefix: /usr/local/Cellar/llvm/9.0.0_1/

    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png~powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab~vsx~vtk+zlib
      version:
      - 4.1.0

    openmpi:
      buildable: False
      version:
      - 4.0
      externals:
      - spec: openmpi@4.0 arch=darwin-mojave-skylake
        prefix: /usr/local/

    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version:
      - 3.7.2
EOF
)
