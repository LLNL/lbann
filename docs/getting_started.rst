Getting Started
=================================

Download
---------------------------------
LBANN can be cloned from the `Github repo 
<https://github.com/LLNL/lbann>`_.
  

Building LBANN
---------------------------------
The build process for LBANN differs on a machine to machine basis. This section describes the build process for LC resources, Spack, and OSX. For users attempting to build on systems not listed above, refer to the LBANN dependencies subsection. 

Livermore Computing build script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building on LC systems is supported via a build script. To run this script navigate to the LBANN directory and run ``build_lbann_lc.sh`` located in ``scripts/``. You will find the resulting executable in ``$LBANN_ROOT/build/<build_information>/install/bin/lbann``

 
OSX build script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building on OSX systems is supported via a build script. To run this script navigate to the LBANN directory and run ``build_lbann_osx.sh`` located in ``scripts/``. You will find the resulting executable in ``$LBANN_ROOT/build/<build_information>/install/bin/lbann``

Building with Spack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LBANN is an available package in `Spack <https://github.com/spack/spack>`_. . To build and install LBANN with spack just  run ``spack install lbann``. For those who intend to modify LBANN's source code the ``spack setup lbann`` command can be used to build LBANN's dependencies. Spack setup creates a configuration file with the dependency locations needed for LBANN's cmake build system. LBANN is built by invoking this script with the path to lbann source, then simply running make.

LBANN CMake and dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LBANN requires the following to build: 

- CMake
- MPI
- Elemental
- OpenCV
- CuDA (optional)
- cuDNN (optional)
- Protocol Buffers (optional)
- Doxygen (optional)

Users comfortable with CMake can choose to build these dependencies themselves, and pass them to LBANN's cmake build system.
