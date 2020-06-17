#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi: [mvapich2@2.3 arch=linux-rhel7-broadwell]
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
        cmake@3.14.5 arch=linux-rhel7-broadwell:  /usr/tce/packages/cmake/cmake-3.14.5

    cuda::
      buildable: False
      version: [10.1.168]
      modules:
        cuda@10.1.168 arch=linux-rhel7-broadwell: cuda/10.1.168

    cudnn::
      buildable: true
      version: [7.6.5.32-10.1-linux-x64]

    gcc::
       buildable: False
       version: [7.3.0]
       modules:
         gcc@7.3.0 arch=linux-rhel7-broadwell: gcc/7.3.0

    hwloc::
      buildable: False
      version: [2.0.2]
      paths:
        hwloc@2.0.2 arch=linux-rhel7-broadwell: /usr/lib64/libhwloc.so

    mvapich2::
      buildable: True
      version: [2.3]
      paths:
        mvapich2@2.3%gcc@7.3.0 arch=linux-rhel7-broadwell: /usr/tce/packages/mvapich2/mvapich2-2.3-gcc-7.3.0/

    openblas::
      buildable: True
      variants: threads=openmp
      version: [0.3.6]

    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png~powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab~vsx~vtk+zlib
      version: [4.1.0]

    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version: [3.7.2]
      modules:
        python@3.7.2 arch=linux-rhel7-broadwell: python/3.7.2

    rdma-core::
      buildable: False
      version: [20]
      paths:
        rdma-core@20 arch=linux-rhel7-broadwell: /usr
EOF
)
