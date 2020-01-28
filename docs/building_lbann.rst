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

With Spack setup and installed into your path, it can be used to
install the LBANN executables. This approach is appropriate for users
that want to train new or existing models using the python front-end.

.. note:: If your model requires custom layers or data readers, you
          may need to install LBANN as a developer, which would allow
          you to modify and recompile the source code.

- Using the Spack environment method:

    .. note:: This method provides a consistent set of dependencies during
        installation.

    .. code-block:: bash

        <path lbann repo>/scripts/install_lbann_from_github.sh -e lbann
        spack env activate -p lbann

    + Options exist in the script to disable the GPUs and change the
      name of the spack environment.

There are numerous options for the spack packages. These options
can be viewed via commands such as :bash:`spack info lbann`. To
specify the compiler, one can add options such as :code:`%gcc@7.3.0`.
For further information about specifying dependencies, such as the MPI
library, please consult `the Spack documentation
<https://spack.readthedocs.io>`_.

.. _build_lbann_from_source:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a developer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developers of LBANN will often need to interact with the source code
and/or advanced configuration options for Aluminum, Hydrogen, and
LBANN while the other dependencies remain constant. The installation
instructions below provide a script that will setup a Spack
environment with the remaining dependencies, and then invoke the LBANN
superbuild CMake infrastructure to build LBANN from the local source
and to download versions of Aluminum and Hydrogen from GitHub.  The
provided script will build with a standard compiler for a given
platform and the nominal options in the CMake build environment.
Expert developers should refer to :ref:`here
<building-with-the-superbuild>` for a list and descriptions of all
CMake flags known to LBANN's "Superbuild" build system.

1.  Running this integrated script that will setup both the
    dependencies via spack and Aluminum, Hydrogen, and LBANN via
    CMake.

    .. code-block:: bash

        <path to lbann repo>/scripts/build_lbann_from_source.sh -p <insert build and install prefix>


    + Options exist in the script to disable the GPUs, and separately
      set the build and install directories.  Additionally, the script
      can setup the environment variables and load the modules in the
      current shell with the following command

    .. code-block:: bash

        source <lbann.git>/scripts/build_lbann.sh -p <build and install prefix>


    + Note that the environments provided here have a set of external
      packages and compilers that are installed on an LLNL LC CZ,
      NERSC, or LLNL configured OS X system.  Please update these for
      your system environment.  Alternatively, you can create baseline
      versions of the user-level Spack configuration files and remove
      the externals and compilers from the :code:`spack.yaml`
      file. More details are provided :ref:`here <setup-spack-env>`.

    + Note that the initial build of all of the standard packages in Spack
      will take a while.


2.  Once the installation has completed you can load the module file
    for LBANN with the following command

    Build LBANN locally from source and build Hydrogen and Aluminum
    using the superbuild. See :ref:`here <building-with-the-superbuild>`
    for a list and descriptions of all CMake flags known to LBANN's
    "Superbuild" build system. A representative CMake command line
    that expects :bash:`LBANN_HOME`, :bash:`LBANN_BUILD_DIR`,
    :bash:`LBANN_INSTALL_DIR` environment variables might be:

    .. code-block:: console

        ml use <path to installation>/etc/modulefiles/
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
