.. role:: bash(code)
          :language: bash

====================
Building LBANN
====================

--------------------
Download
--------------------

LBANN source code can be obtained from the `Github
repo <https://github.com/LLNL/lbann>`_.

.. _building-with-spack:

------------------------------------------------------------
Building with `Spack <https://github.com/llnl/spack>`_
------------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup Spack and local base tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Download and install `Spack <https://github.com/llnl/spack>`_.
    Additionally setup shell support as discussed
    `here <https://spack.readthedocs.io/en/latest/module_file_support.html#id2>`_.

    .. code-block:: bash

        source ${SPACK_ROOT}/share/spack/setup-env.sh


2.  Setup your compiler and external software environment. For example,
    on LLNL\'s LC machines, one might load the following modules:

    .. code-block:: bash

        ml gcc/7.3.0 mvapich2/2.3 cuda/10.0.130 # Pascal

    or

    .. code-block:: bash

        ml gcc/7.3.1 cuda/9.2.148 spectrum-mpi/rolling-release  # Lassen / Sierra


    + Note to unload unwanted modules you can execute :bash:`ml` with
      package names prepended with a dash, e.g.: :bash:`ml -intel`. To
      unload all currently loaded modules, use :bash:`ml purge`.

3.  Optionally, setup your spack environment to take advantage of
    locally installed tools.  Note that unless your spack environment
    is explicitly told about tools such as cmake, python, mpi, etc. it
    will install everything that LBANN and all of its dependencies
    require. This can take quite a long time, but only has to be done
    once for a given spack repository.  Once all of the standard tools
    are installed, rebuilding LBANN with spack is quite fast.

    + Advice on setting up paths to external installations is beyond
      the scope of this document, but is covered in the `Spack
      Documentation <https://spack.readthedocs.io/>`_.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a user
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: This section is still under development and being
             tested. It contains known issues. This warning will be
             removed when it is believed to be generally usable.

With Spack setup and installed into your path, it can be used to
install the LBANN executables. This approach is appropriate for users
that want to train new or existing models using the python front-end.

.. note:: If your model requires custom layers or data readers, you
          may need to install LBANN as a developer, which would allow
          you to modify and recompile the source code.

Here are three easy ways to install LBANN:

- Using the Spack environment method, (e.g., for an x86_64 LLNL LC
  system with GPU support):

  .. note:: This method provides a consistent set of dependencies during
      installation.

  .. code-block:: bash

      cd <path to LBANN repo>/spack_environments/users/llnl_lc/<arch>_cuda/ # where <arch> = x86_64 | ppc64le
      spack install
      spack env loads
      source ./loads

- Building with the latest released versions and GPU support (use the
  user's defaults for specifying the compiler, MPI library, etc.):

  .. code-block:: bash

      spack install lbann +gpu +nccl
      ml load lbann

- Building with the head of develop branch for lbann, hydrogen and
  aluminum with GPU support (use the user's defaults for specifying
  the compiler, MPI library, etc.):

  .. code-block:: bash

      spack install lbann@develop +gpu +nccl ^hydrogen@develop ^aluminum@master
      ml load lbann

There are numerous options for all of these packages. These options
can be viewed via commands such as :bash:`spack info lbann`. To
specify the compiler, one can add options such as :code:`%gcc@7.3.0`.
For further information about specifying dependencies, such as the MPI
library, please consult `the Spack documentation
<https://spack.readthedocs.io>`_.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a developer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developers of LBANN will often need to interact with the source code
and/or advanced configuration options for Aluminum, Hydrogen, and
LBANN while the other dependencies remain constant. The Spack
installation instructions below set up a Spack environment with the
remaining dependencies, requiring the developer to build Aluminum,
Hydrogen, and LBANN separately, by whatever means they choose.

1.  Establish a Spack environment and install software dependencies.
    Note that there are four environments to pick from along two axes:

    .. note:: This spack environment has to be setup once each time
              you create a new build directory.

    1. developers or users
    2. x86_64 and ppc64le

    For example if you are a developer and want to build the inside of
    the git repo use the following instructions:

    .. code-block:: bash

        export LBANN_HOME=/path/to/lbann/git/repo
        export LBANN_BUILD_DIR=/path/to/a/build/directory
        export LBANN_INSTALL_DIR=/path/to/an/install/directory
        cd ${LBANN_BUILD_DIR}
        spack env create -d . ${LBANN_HOME}/spack_environments/developer_release_<arch>_cuda_spack.yaml # where <arch> = x86_64 | ppc64le
        cp ${LBANN_HOME}/spack_environments/std_versions_and_variants_llnl_lc_cz.yaml .
        cp ${LBANN_HOME}/spack_environments/externals_<arch>_llnl_lc_cz.yaml . # where <arch> = x86_64 | ppc64le
        spack install
        spack env loads # Spack creates a file named loads that has all of the correct modules
        source ${SPACK_ROOT}/share/spack/setup-env.sh # Rerun setup since spack doesn't modify MODULEPATH unless there are module files defined
        source loads


    + Note that the environments provided here have a set of external
      packages and compilers that are installed on an LLNL LC CZ
      system.  Please update these for your system environment.
      Alternatively, you can create baseline versions of the
      user-level Spack configuration files and remove the externals
      and compilers from the :code:`spack.yaml` file. More details are
      provided :ref:`here <setup-spack-env>`.

    + Note that the initial build of all of the standard packages in Spack
      will take a while.

    + Note that the Spack module files set the :bash:`LIBRARY_PATH` environment
      variable. This behavior allows autotools-based builds to pickup the
      correct libraries but interferes with the way that CMake sets up
      RPATHs.  To correctly establish the RPATH, please unset the variable
      as noted above, or you can explicitly pass the RPATH fields to CMake
      using a command such as:

      .. code-block:: bash

          cmake -DCMAKE_INSTALL_RPATH=$(sed 's/:/;/g' <<< "${LIBRARY_PATH}") \
                -DCMAKE_BUILD_RPATH=$(sed 's/:/;/g' <<< "${LIBRARY_PATH}") \
                ...

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
          -D ALUMINUM_ENABLE_NCCL=ON \
          \
          -D LBANN_SB_BUILD_HYDROGEN=ON \
          -D Hydrogen_ENABLE_ALUMINUM=ON \
          -D Hydrogen_ENABLE_CUB=ON \
          -D Hydrogen_ENABLE_CUDA=ON \
          \
          -D LBANN_SB_BUILD_LBANN=ON \
          -D LBANN_DATATYPE:STRING=float \
          -D LBANN_SEQUENTIAL_INITIALIZATION:BOOL=OFF \
          -D LBANN_WITH_ALUMINUM:BOOL=ON \
          -D LBANN_WITH_CONDUIT:BOOL=ON \
          -D LBANN_WITH_CUDA:BOOL=ON \
          -D LBANN_WITH_CUDNN:BOOL=ON \
          -D LBANN_WITH_NCCL:BOOL=ON \
          -D LBANN_WITH_NVPROF:BOOL=ON \
          -D LBANN_WITH_SOFTMAX_CUDA:BOOL=ON \
          -D LBANN_WITH_TOPO_AWARE:BOOL=ON \
          -D LBANN_WITH_TBINF=OFF \
          -D LBANN_WITH_VTUNE:BOOL=OFF \
          ${LBANN_HOME}/superbuild

        ninja
        ml use ${LBANN_INSTALL_DIR}/etc/modulefiles/
        ml load lbann-0.99.0


The complete documentation for building LBANN directly with CMake can
be found :ref:`here <build-with-cmake>`.

------------------------------
Advanced build methods
------------------------------

.. toctree::
   :maxdepth: 1

   build_osx
   build_with_cmake
   build_with_superbuild
   build_containers
   build_llnl_idiosyncracies
   build_spack_extra_config
