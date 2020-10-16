#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi:
          - spectrum-mpi@rolling-release arch=linux-rhel7-power9le
        lapack:
          - openblas threads=openmp
        blas:
          - openblas threasd=openmp
      buildable: true
      version: []
EOF
)

EXTERNAL_PACKAGES=$(cat <<EOF
    cmake::
      buildable: True
      variants: ~openssl ~ncurses
      version: [3.18.0]
      externals:
      - spec: cmake@3.18.0 arch=linux-rhel7-power9le
        modules:
        - cmake/3.18.0
    cuda::
      buildable: False
      version: [11.0.2]
      externals:
      - spec: cuda@11.0.2 arch=linux-rhel7-power9le
        modules:
        - cuda/11.0.2
    cudnn::
      buildable: true
      version:
      - 8.0.4.30-11.0-linux-ppc64le
    gcc::
      buildable: False
      version:
      - 8.3.1
      externals:
      - spec: gcc@8.3.1 arch=linux-rhel7-power9le
        modules:
        - gcc/8.3.1
    hwloc::
      buildable: False
      version:
      - 2.0.2
      externals:
      - spec: hwloc@2.0.2 arch=linux-rhel7-power9le
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
    perl::
      buildable: False
      version:
        - 5.16.3
      externals:
      - spec: perl@5.16.3 arch=linux-rhel7-power9le
        prefix: /usr
    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version:
      - 3.7.2
      externals:
      - spec: python@3.7.2 arch=linux-rhel7-power9le
        modules:
        - python/3.7.2
    rdma-core::
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=linux-rhel7-power9le
        prefix: /usr
    spectrum-mpi::
      buildable: False
      version:
      - rolling-release
      externals:
      - spec: spectrum-mpi@rolling-release %gcc@8.3.1 arch=linux-rhel7-power9le
        prefix: /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1
EOF
)
