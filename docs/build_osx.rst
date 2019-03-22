.. role:: bash(code)
          :language: bash

=========================
Building LBANN on OS X
=========================

.. warning:: This section is still under development and being
             tested. It contains known issues. This warning will be
             removed when it is believed to be generally usable.


--------------------
Getting Started
--------------------

.. _osx-setup-spack:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup Spack and local base tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get started follow the general directions on building LBANN to
`setup spack
<https://lbann.readthedocs.io/en/latest/building_lbann.html#setup-spack-and-local-base-tools>`_.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup Homebrew
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: Setting up Homebrew only needs to be done once per system,.

1.  Download and install `Homebrew <https://brew.sh>`_.  Setup base
    development packages.  Note that at the moment we use brew to
    install llvm, open-mpi, scalapack, and cmake.

    .. code-block:: bash

       brew install llvm
       brew install open-mpi
       brew install scalapack
       brew install cmake

    Put the brew based clang in your path:

    .. code-block:: bash

       export PATH="/usr/local/opt/llvm/bin:$PATH";

    Install lmmod so that we can use modules to put spack built
    packages into your path.

    .. code-block:: bash

       brew install lmod
       brew install luarocks

    Update your .profile to enable use of modules via lmod

    .. code-block:: bash

       source $(brew --prefix lmod)/init/$(basename $SHELL)

.. _osx-build-install-as-developer:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a developer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Establish a Spack environment and install software dependencies.

    .. note:: This spack environment has to be setup once each time
              you create a new build directory.

    .. code-block:: bash

        export LBANN_HOME=/path/to/lbann/git/repo
        export LBANN_BUILD_DIR=/path/to/a/build/directory
        export LBANN_INSTALL_DIR=/path/to/an/install/directory
        cd ${LBANN_BUILD_DIR}
        spack env create -d . ${LBANN_HOME}/spack_environments/developer_release_osx_spack.yaml
        spack install
        spack env loads # Spack creates a file named loads that has all of the correct modules
        source loads
        unset LIBRARY_PATH


2.  Build LBANN locally from source and build Hydrogen and Aluminum
    using the superbuild. See :ref:`here <building-with-the-superbuild>`
    for a list and descriptions of all CMake flags known to LBANN's
    "Superbuild" build system. A representative CMake command line
    that expects :bash:`LBANN_HOME`, :bash:`LBANN_BUILD_DIR`,
    :bash:`LBANN_INSTALL_DIR` environment variables might be:

    .. code-block:: console

        cd ${LBANN_BUILD_DIR}
        cmake \
          -G Ninja \
          -D CMAKE_BUILD_TYPE:STRING=Release \
          -D CMAKE_INSTALL_PREFIX:PATH=${LBANN_INSTALL_DIR} \
          \
          -D LBANN_SB_BUILD_ALUMINUM=ON \
          -D ALUMINUM_ENABLE_MPI_CUDA=OFF \
          -D ALUMINUM_ENABLE_NCCL=OFF \
          \
          -D LBANN_SB_BUILD_HYDROGEN=ON \
          -D Hydrogen_ENABLE_ALUMINUM=ON \
          -D Hydrogen_ENABLE_CUB=OFF \
          -D Hydrogen_ENABLE_CUDA=OFF \
          \
          -D LBANN_SB_BUILD_LBANN=ON \
          -D LBANN_DATATYPE:STRING=float \
          -D LBANN_SEQUENTIAL_INITIALIZATION:BOOL=OFF \
          -D LBANN_WITH_ALUMINUM:BOOL=ON \
          -D LBANN_WITH_CONDUIT:BOOL=ON \
          -D LBANN_WITH_CUDA:BOOL=OFF \
          -D LBANN_WITH_CUDNN:BOOL=OFF \
          -D LBANN_WITH_NCCL:BOOL=OFF \
          -D LBANN_WITH_NVPROF:BOOL=OFF \
          -D LBANN_WITH_SOFTMAX_CUDA:BOOL=OFF \
          -D LBANN_WITH_TOPO_AWARE:BOOL=ON \
          -D LBANN_WITH_TBINF=OFF \
          -D LBANN_WITH_VTUNE:BOOL=OFF \
          \
          -D CMAKE_CXX_COMPILER=$(which clang) \
          -D CMAKE_C_COMPILER=$(which clang) \
          -D LBANN_SB_FWD_ALUMINUM_OpenMP_CXX_LIB_NAMES=omp \
          -D LBANN_SB_FWD_ALUMINUM_OpenMP_CXX_FLAGS=-fopenmp \
          -D LBANN_SB_FWD_ALUMINUM_OpenMP_omp_LIBRARY=/usr/local/opt/llvm/lib/libomp.dylib \
          ${LBANN_HOME}/superbuild

        ninja
