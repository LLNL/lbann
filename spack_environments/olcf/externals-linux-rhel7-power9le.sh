#!/bin/sh

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi:
          - spectrum-mpi@10.3.1.2-20200121 arch=linux-rhel7-power9le
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
        - 3.18.0
    cuda::
      buildable: False
      version:
      - 10.1.243
      target: []
      compiler: []
      providers: {}    
      externals:
      - spec: cuda@10.1.243 arch=linux-rhel7-power9le
        modules:
        - cuda/10.1.243
    cudnn::
      buildable: true
      version:
        - 8.0.2.39-10.1-linux-ppc64le
    gcc::
      buildable: False
      version:
      - 7.4.0
      target: []
      compiler: []
      providers: {}    
      externals:
      - spec: gcc@7.4.0 arch=linux-rhel7-power9le
        modules:
        - gcc/7.4.0
    hwloc::
      buildable: False
      version:
      - 2.0.2
      target: []
      compiler: []
      providers: {}    
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
    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version:
        - 3.7.2
    rdma-core::
      buildable: False
      version:
       - 20
      target: []
      compiler: []
      providers: {}    
      externals:
      - spec: rdma-core@20 arch=linux-rhel7-power9le
        prefix: /usr
    spectrum-mpi::
      buildable: False
      version:
      - 10.3.1.2-20200121
      target: []
      compiler: []
      providers: {}    
      externals:
      - spec: spectrum-mpi@10.3.1.2-20200121 %gcc@7.4.0 arch=linux-rhel7-power9le
        modules:
        - spectrum-mpi/10.3.1.2-20200121
EOF
)
