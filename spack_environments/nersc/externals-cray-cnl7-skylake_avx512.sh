#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi: [mvapich2@2.3 arch=cray-cnl7-skylake_avx512]
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
      version: [3.14.4]
      modules:
        cmake@3.14.4 arch=cray-cnl7-skylake_avx512: cmake/3.14.4

    cuda::
      buildable: False
      version: [10.2.89]
      modules:
        cuda@10.2.89 arch=cray-cnl7-skylake_avx512: cuda/10.2.89

    cudnn::
      buildable: true
      version: [7.6.5.32-10.2-linux-x64]

    gcc::
       buildable: False
       version: [8.2.0]
       modules:
         gcc@8.2.0 arch=cray-cnl7-skylake_avx512: gcc/8.2.0

    gettext::
      buildable: False
      version: [0.19.8.1]
      paths:
        gettext@0.19.8.1 arch=cray-cnl7-skylake_avx512: /usr

    hwloc::
      buildable: False
      version: [1.11.8]
      paths:
        hwloc@1.11.8 arch=cray-cnl7-skylake_avx512: /usr/lib64/libhwloc.so

    mvapich2::
      buildable: False
      version: [2.3.2]
      modules:
        mvapich2@2.3.2%gcc@8.2.0 arch=cray-cnl7-skylake_avx512: mvapich2/2.3.2

    openblas::
      buildable: True
      variants: threads=openmp
      version: [0.3.10]

    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png~powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab~vsx~vtk+zlib
      version: [4.1.0]

    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version: [3.7.4]

    readline::
      buildable: False
      version: [8.0]
      paths:
        readline@8.0 arch=cray-cnl7-skylake_avx512: /lib64

    rdma-core::
      buildable: False
      version: [20]
      paths:
        rdma-core@20 arch=cray-cnl7-skylake_avx512: /usr
EOF
)
