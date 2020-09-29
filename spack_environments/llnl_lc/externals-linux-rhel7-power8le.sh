#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi:
          - spectrum-mpi@rolling-release arch=linux-rhel7-power8le
        lapack:
          - openblas threads=openmp
        blas:
          - openblas threads=openmp
      buildable: true
      version: []
EOF
)

EXTERNAL_PACKAGES=$(cat <<EOF
    cmake::
      buildable: True
      variants: ~openssl ~ncurses
      version:
      - 3.14.5
      externals:
      - spec: cmake@3.14.5 arch=linux-rhel7-power8le
        modules:
        - cmake/3.14.5
    cuda::
      buildable: False
      version:
      - 10.2.89
      externals:
      - spec: cuda@10.1.168 arch=linux-rhel7-power8le
        modules:
        - cuda/10.1.168
    cudnn::
      buildable: true
      version:
      - 7.6.5.32-10.1-linux-ppc64le
    gcc::
      buildable: False
      version:
      - 7.3.1
      externals:
      - spec:  gcc@7.3.1 arch=linux-rhel7-power8le
        modules:
        - gcc/7.3.1
    hwloc::
      buildable: False
      version:
      - 2.0.2
      externals:
      - spec: hwloc@2.0.2 arch=linux-rhel7-power8le
        prefix: /usr/lib64/libhwloc.so
    openblas::
      buildable: True
      variants: threads=openmp ~avx2 ~avx512
      version:
      - 0.3.10
    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png+powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab+vsx~vtk+zlib
      version:
      - 4.1.0
    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version:
      - 3.7.2
      externals:
      - spec: python@3.7.2 arch=linux-rhel7-power8le
        modules:
        - python/3.7.2
    rdma-core::
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=linux-rhel7-power8le
        prefix: /usr
    spectrum-mpi::
      buildable: False
      version:
      - rolling-release
      externals:
      - spec: spectrum-mpi@rolling-release %gcc@7.3.1 arch=linux-rhel7-power8le
        prefix: /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-7.3.1
EOF
)
