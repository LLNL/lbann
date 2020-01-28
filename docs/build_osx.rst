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

.. warning:: If using OSX 10.14 or newer, be sure that
             :bash:`/usr/include` has been restored. In version 10.14,
             this may be accomplished by installing
             :bash:`/Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg`.
             If this package is not available, it's possible command
             line tools have not been installed; do so by executing
             :bash:`xcode-select --install`.


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
       brew install hwloc

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


2.  Follow the main directions for developers listed :ref:`here
    <build_lbann_from_source>`.  This will build LBANN locally from
    source, Hydrogen and Aluminum using the superbuild, and everything
    else from Spack.  See :ref:`here <building-with-the-superbuild>`
    for a list and descriptions of all CMake flags known to LBANN's
    "Superbuild" build system.
