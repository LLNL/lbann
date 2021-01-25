#!/bin/sh

if [[ ${GPU_ARCH_VARIANTS} ]]; then
    GLOBAL_GPU_ARCH_VARIANTS="variants: ${GPU_ARCH_VARIANTS}"
fi

EXTERNAL_ALL_PACKAGES=$(cat <<EOF
    all:
      providers:
        mpi:
          - mvapich2@2.3 arch=${SPACK_ARCH}
        lapack:
          - openblas threads=openmp
        blas:
          - openblas threasd=openmp
      buildable: true
      version: []
      ${GLOBAL_GPU_ARCH_VARIANTS}
EOF
)

EXTERNAL_PACKAGES=$(cat <<EOF
    cmake::
      buildable: True
      variants: ~openssl ~ncurses
      version:
      - 3.18.0
      externals:
      - spec: cmake@3.18.0 arch=${SPACK_ARCH}
        modules:
        -  cmake/3.18.0
    cuda::
      buildable: False
      version:
      - 11.1.0
      externals:
      - spec: cuda@11.1.0 arch=${SPACK_ARCH}
        modules:
        - cuda/11.1.0
    cudnn::
      buildable: true
      version:
      - 8.0.4.30-11.1
    gcc::
      buildable: False
      version:
      - 8.3.1
      externals:
      - spec:  gcc@8.3.1 arch=${SPACK_ARCH}
        modules:
        - gcc/8.3.1
    hwloc::
      buildable: False
      version:
      - 1.11.13
      variants: +cuda +nvml
      externals:
      - spec: hwloc@1.11.13 arch=${SPACK_ARCH}
        prefix: /usr/lib64/libhwloc.so
    mvapich2::
      buildable: True
      version:
      - 2.3
      externals:
      - spec: mvapich2@2.3%gcc@8.3.1 arch=${SPACK_ARCH}
        prefix: /usr/tce/packages/mvapich2/mvapich2-2.3-gcc-8.3.1/
    openblas::
      buildable: True
      variants: threads=openmp
      version:
      - 0.3.10
    opencv::
      buildable: true
      variants: build_type=RelWithDebInfo ~calib3d+core~cuda~dnn~eigen+fast-math~features2d~flann~gtk+highgui+imgcodecs+imgproc~ipp~ipp_iw~jasper~java+jpeg~lapack~ml~opencl~opencl_svm~openclamdblas~openclamdfft~openmp+png~powerpc~pthreads_pf~python~qt+shared~stitching~superres+tiff~ts~video~videoio~videostab~vsx~vtk+zlib
      version:
      - 4.1.0
    perl::
      buildable: False
      version:
        - 5.16.3
      externals:
      - spec: perl@5.16.3 arch=${SPACK_ARCH}
        prefix: /usr
    python::
      buildable: True
      variants: +shared ~readline ~zlib ~bz2 ~lzma ~pyexpat
      version:
      - 3.7.2
      externals:
      - spec: python@3.7.2 arch=${SPACK_ARCH}
        modules:
        - python/3.7.2
    rdma-core::
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${SPACK_ARCH}
        prefix: /usr
EOF
)
