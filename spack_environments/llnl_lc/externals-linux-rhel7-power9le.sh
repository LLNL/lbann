#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi: [spectrum-mpi@rolling-release arch=linux-rhel7-power9le]
        lapack: [openblas threads=openmp]
        blas: [openblas threasd=openmp]
      buildable: true
      version: []
      paths: {}
      modules: {}
EOF
)

EXTERNAL_PACKAGES=$(cat <<EOF
    cmake::
      buildable: True
      variants: ~openssl ~ncurses
      version: [3.14.5]
      paths:
        cmake@3.14.5 arch=linux-rhel7-power9le:   /usr/tce/packages/cmake/cmake-3.14.5

    cuda::
      buildable: False
      version: [9.2.88, 10.1.105, 10.1.168, 10.1.243]
      paths:
        cuda@9.2.88 arch=linux-rhel7-power9le: /usr/tce/packages/cuda/cuda-9.2.88/
        cuda@10.1.105 arch=linux-rhel7-power9le: /usr/tce/packages/cuda/cuda-10.1.105
        cuda@10.1.168 arch=linux-rhel7-power9le: /usr/tce/packages/cuda/cuda-10.1.168
        cuda@10.1.243 arch=linux-rhel7-power9le: /usr/tce/packages/cuda/cuda-10.1.243

    cudnn::
      buildable: true
      version: [7.4.2. 7.5.1, 7.5.1-10.1-power9le, 7.6.3-10.1-power9le]
      paths:
        cudnn@7.5.1 arch=linux-rhel7-power9le: /usr/workspace/wsb/brain/cudnn/cudnn-7.5.1/cuda-10.1_ppc64le/
        cudnn@7.4.2 arch=linux-rhel7-power9le: /usr/workspace/wsb/brain/cudnn/cudnn-7.4.2/cuda-9.2_ppc64le/

    gcc::
       buildable: False
       version: [7.3.1]
       modules:
         gcc@7.3.1 arch=linux-rhel7-power9le: gcc/7.3.1

    hwloc::
      buildable: False
      version: [2.0.2]
      paths:
        hwloc@2.0.2 arch=linux-rhel7-power9le: /usr/lib64/libhwloc.so

    openblas::
      buildable: True
      variants: threads=openmp ~avx2 ~avx512
      version: [0.3.6]

    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png+powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab+vsx~vtk+zlib
      version: [4.1.0]

    spectrum-mpi::
      buildable: False
      version: [rolling-release]
      paths:
        spectrum-mpi@rolling-release %gcc@7.3.1 arch=linux-rhel7-power9le: /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-7.3.1
EOF
)
