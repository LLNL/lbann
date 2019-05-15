.. _setup-spack-env:

========================================
Setting up Basic Spack Environment
========================================

.. note:: These instructions are specific to LLNL's Livermore
          Computing (LC) machines x86_64 machines. External users will
          likely have to modify these paths to be specific to their
          build platform.

+ Copy the following text into :code:`~/.spack/linux/compilers.yaml`: ::

        compilers:
        - compiler:
            environment: {}
            extra_rpaths: []
            flags: {}
            modules: []
            operating_system: rhel7
            paths:
              cc: /usr/tce/packages/gcc/gcc-7.3.0/bin/gcc
              cxx: /usr/tce/packages/gcc/gcc-7.3.0/bin/g++
              f77: /usr/tce/packages/gcc/gcc-7.3.0/bin/gfortran
              fc: /usr/tce/packages/gcc/gcc-7.3.0/bin/gfortran
            spec: gcc@7.3.0
            target: x86_64
        - compiler:
            environment: {}
            extra_rpaths: []
            flags: {}
            modules: []
            operating_system: rhel7
            paths:
              cc: /usr/tce/packages/gcc/gcc-7.3.1/bin/gcc
              cxx: /usr/tce/packages/gcc/gcc-7.3.1/bin/g++
              f77: /usr/tce/packages/gcc/gcc-7.3.1/bin/gfortran
              fc: /usr/tce/packages/gcc/gcc-7.3.1/bin/gfortran
            spec: gcc@7.3.1
            target: ppc64le

+ Copy the following text into :code:`~/.spack/linux/packages.yaml`: ::

        packages:
          all:
            compiler: [gcc]

          cmake:
            variants: ~openssl ~ncurses
            paths:
              cmake@3.12.1 arch=linux-rhel7-x86_64:  /usr/tce/packages/cmake/cmake-3.12.1

          mvapich2:
            buildable: True
            version: [2.3]
            paths:
              mvapich2@2.3%gcc@7.3.0 arch=linux-rhel7-x86_64: /usr/tce/packages/mvapich2/mvapich2-2.3-gcc-7.3.0/

          hwloc:
            buildable: False
            version: [2.0.2]
            paths:
              hwloc@2.0.2 arch=linux-rhel7-x86_64: /usr/lib64/libhwloc.so

          cuda:
            buildable: False
            version: [9.2.88, 10.0.130]
            paths:
              cuda@10.0.130 arch=linux-rhel7-x86_64: /usr/tce/packages/cuda/cuda-10.0.130
              cuda@9.2.88 arch=linux-rhel-ppc64le: /usr/tce/packages/cuda/cuda-9.2.88/

          cudnn:
            buildable: False
            version: [7.4.2]
            paths:
              cudnn@7.4.2 arch=linux-rhel7-x84_64: /usr/workspace/wsb/brain/cudnn/cudnn-7.4.2/cuda-10.0_x86_64
              cudnn@7.4.2 arch=linux-rhel-ppc64le: /usr/workspace/wsb/brain/cudnn/cudnn-7.4.2/cuda-9.2_ppcle

          nccl:
            buildable: False
            version: [2.3]
            paths:
              nccl@2.3 arch=linux-rhel7-x84_64: /usr/workspace/wsb/brain/nccl2/nccl_2.3.7-1+cuda10.0_x86_64

          spectrum-mpi:
            buildable: False
            version: [rolling-release]
            paths:
              spectrum-mpi@rolling-release %gcc@7.3.1 arch=linux-rhel-ppc64le: /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-7.3.1
