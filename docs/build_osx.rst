.. role:: bash(code)
          :language: bash

=========================
Building LBANN on OS X
=========================

.. warning:: If using OSX 10.14 or newer, be sure that
             :bash:`/usr/include` has been restored. In version 10.14,
             this may be accomplished by installing
             :bash:`/Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg`.
             If this package is not available, it's possible command
             line tools have not been installed; do so by executing
             :bash:`xcode-select --install`.


.. _osx-basic-setup:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup Homebrew
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: Setting up Homebrew only needs to be done once per system,.

Download and install `Homebrew <https://brew.sh>`_.  Setup base
development packages. Note that at the moment we use brew to install
LLVM, Open-MPI, ScaLAPACK, and CMake.

.. code-block:: bash

   brew install llvm
   brew install open-mpi
   brew install cmake
   brew install hwloc

Put the brew-based :code:`clang` in your path:

.. code-block:: bash

   export PATH=/usr/local/opt/llvm/bin:$PATH;

Install :code:`lmod` so that we can use modules to put Spack-built
packages into your path:

.. code-block:: bash

   brew install lmod
   brew install luarocks

Update your shell configuration files to enable use of modules via
:code:`lmod`:

.. code-block:: bash

   source $(brew --prefix lmod)/init/$(basename $SHELL)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From this point, follow the instructions for :ref:`building LBANN with
Spack <building-with-spack>`.
