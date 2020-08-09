#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi: [spectrum-mpi@10.3.1.2-20200121 arch=linux-rhel7-power9le]
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
      version: [3.18.0]
#      modules:
#        cmake@3.17.3 arch=linux-rhel7-power9le: cmake/3.17.3

    cuda::
      buildable: False
      version: [11.0.2]
#      version: [10.1.243]
      modules:
        cuda@11.0.2 arch=linux-rhel7-power9le: cuda/11.0.2
#        cuda@10.1.243 arch=linux-rhel7-power9le: cuda/10.1.243

    cudnn::
      buildable: true
      version: [8.0.2.39-11.0-linux-ppc64le]

    gcc::
       buildable: False
       version: [8.1.1]
       modules:
         gcc@8.1.1 arch=linux-rhel7-power9le: gcc/8.1.1

    hwloc::
      buildable: False
      version: [2.0.2]
      paths:
        hwloc@2.0.2 arch=linux-rhel7-power9le: /usr/lib64/libhwloc.so

    openblas::
      buildable: True
      variants: threads=openmp ~avx2 ~avx512
      version: [0.3.10]

    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png+powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab+vsx~vtk+zlib
      version: [4.1.0]

    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version: [3.7.2]
#      modules:
#        python@3.7.0 arch=linux-rhel7-power9le: python/3.7.0

    rdma-core::
      buildable: False
      version: [20]
      paths:
        rdma-core@20 arch=linux-rhel7-power9le: /usr

    spectrum-mpi::
      buildable: False
      version: [10.3.1.2-20200121]
      modules:
        spectrum-mpi@10.3.1.2-20200121 %gcc@8.1.1 arch=linux-rhel7-power9le: spectrum-mpi/10.3.1.2-20200121
#        spectrum-mpi@10.3.1.2-20200121 %gcc@7.4.0 arch=linux-rhel7-power9le: spectrum-mpi/10.3.1.2-20200121
EOF
)
