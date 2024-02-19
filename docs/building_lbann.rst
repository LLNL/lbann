 .. role:: bash(code)
          :language: bash

====================
Building LBANN
====================

--------------------
Download
--------------------

LBANN source code can be obtained from the `Github
repository <https://github.com/LLNL/lbann>`_.

.. _building-with-spack:

------------------------------------------------------------
Building with `Spack <https://github.com/llnl/spack>`_
------------------------------------------------------------

.. note:: Users attempting to install LBANN on a Mac OSX machine may
          need to do :ref:`additional setup <osx-basic-setup>` before
          continuing. In particular, installing LBANN requires a
          different compiler than the default OSX command line tools
          and an MPI library.

.. _setup_spack:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup Spack (One-time setup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Download and install `Spack <https://github.com/llnl/spack>`_
    using the following commands.
    Additionally setup shell support as discussed
    `here <https://spack.readthedocs.io/en/latest/module_file_support.html#id2>`_.

    .. code-block:: bash

        git clone https://github.com/spack/spack.git spack.git
        export SPACK_ROOT=<path to installation>/spack.git
        source ${SPACK_ROOT}/share/spack/setup-env.sh

2.  LBANN will use `Spack environments
    <https://spack.readthedocs.io/en/latest/environments.html>`_ to
    specify and manage both compilers and versions of dependent
    libraries.  Go to the install instructions for :ref:`users
    <install_lbann_as_user>` or :ref:`developers
    <build_lbann_from_source>`.

.. note:: Optionally, setup your Spack environment to take advantage
          of locally installed tools.  Unless your Spack environment
          is explicitly told about tools such as CMake, Python, MPI,
          etc., it will install everything that LBANN and all of its
          dependencies require. This can take quite a long time but
          only has to be done once for a given spack repository. Once
          all of the standard tools are installed, rebuilding LBANN
          with Spack is quite fast.

          Advice on setting up paths to external installations is
          beyond the scope of this document but is covered in the
          `Spack Documentation
          <https://spack.readthedocs.io/en/latest/configuration.html>`_.


.. _install_lbann_as_user_known_center:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a user at a known HPC Center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Spack setup and installed into your path, it can be used to
install the LBANN executables. This approach is appropriate for users
that want to train new or existing models using the Python front-end
and are working at known HPC Computing Centers (e.g. LLNL LC, OLCF,
ALCF, RIKEN).  The scripts that are invoked load known modules to tie
into system install packages (e.g. MPI).

1. Execute LBANN's user build script:

   .. code-block:: bash

      /bin/bash -c "$(curl -fsSL https://github.com/LLNL/lbann/raw/develop/scripts/.build_lbann_as_user.sh)"

   .. note::

      There are three environment variables that can be used to
      override the default:
        - version of LBANN being installed (LBANN_VERSION),
        - the variants enabled (LBANN_VARIANTS),
        - and extra packages to build in the same environment (LBANN_EXTRAS).
      The following is an example of Using the variables:

      .. code-block:: bash

         LBANN_VERSION="develop" \
         LBANN_VARIANTS="+cuda +numpy" \
         LBANN_EXTRAS="-p py-torch@1.7.1 -p py-numpy" \
         /bin/bash -c "$(curl -fsSL https://github.com/LLNL/lbann/raw/develop/scripts/.build_lbann_as_user.sh)"

2. Once this is done, you can load LBANN by activating the spack environment:

   .. code-block:: bash

      spack env activate -p lbann-user-broadwell

.. _install_lbann_as_user:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a user (with maximum flexibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Spack setup and installed into your path, it can be used to
install the LBANN executables. This approach is appropriate for users
that want to train new or existing models using the Python front-end.

.. note:: Users should make themselves comfortable with Spack and `its
          idioms for installing packages
          <https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html>`_,
          and experts can add `customizations to their Spack ecosystem
          <https://spack.readthedocs.io/en/latest/configuration.html>`_
          (and modify these instructions) to ensure that they get the
          compilers and externals that they want.

.. _setting_up_clingo:

.. note:: LBANN works best with Spack's new concretizer clingo.
          Please enable it by performing the following steps.  This
          only needs to be done once per spack repository.

          To update spack to use the new clingo concretizer.  There are two
          steps to this: bootstrapping clingo and modifying the repositories
          configuration file.

          .. code-block:: bash

             module load gcc/8.3.1     # Load a compiler with C++14 support
             spack compiler add        # Make Spack aware of the new compiler
             spack solve zlib          # Force Spack to bootstrap clingo

          Create the file :code:`${SPACK_ROOT}/etc/spack/config.yaml`
          if it doesn't exist. Verify that this file contains the
          following lines, adding them if necessary:

          .. code-block:: bash

             config:
               concretizer: clingo

.. note:: If your model requires custom layers or data readers, you
          may need to install LBANN as a developer, which would allow
          you to modify and recompile the source code.

The best practices are to create a Spack environment (similar to a
Python virtual environment) and to use something like your compute
center's :code:`modules` packages to provide paths to system installed
software.

1. Create and activate spack environment called (replace :code:`<env name>`):

.. code-block:: bash

   spack env create <env name>
   spack env activate -p <env name>

2. Load relevant modules and find pre-installed software:

.. code-block:: bash

   module load <modules that you want>
   spack compiler find --scope env:<env name>
   spack external find --scope env:<env name>

3. Install LBANN with the variants that you care about:

.. code-block:: bash

   spack install --add lbann <variants and dependencies>
   spack load lbann@develop

.. note::

   Here is an example of a set of commands that works on an x86_64
   architecture with Nvidia P100 GPUs:

   .. code-block:: bash

      spack env create lbann
      spack env activate -p lbann
      module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.0 mvapich2/2.3 python/3.7.2
      spack compiler find --scope env:lbann
      spack external find --scope env:lbann
      spack install --add lbann@develop cuda_arch=60 +cuda ^hydrogen@develop+al ^aluminum@master ^mvapich2
      spack load lbann@develop

Please note that when getting LBANN to build as a user will encounter
some issues with the Spack legacy concretizer and use of the new
clingo concretizer is highly recommended :ref:`(see above)
<setting_up_clingo>`. Using the legacy concretizer will require getting
just the "right" invocation and we suggest using clingo.  Furthermore,
it is typically best to have Spack reuse a version of MPI installed by
the system administrators to ensure that it is properly configured.

.. _build_lbann_from_source:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a developer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developers of LBANN will often need to interact with the source code
and/or set advanced configuration options for Aluminum, Hydrogen, and
LBANN while the other dependencies remain constant. The installation
instructions below provide a script that will setup a Spack
environment with the remaining dependencies, and then invoke the LBANN
CMake infrastructure to build LBANN from the local source. The
provided script will build with a standard compiler for a given
platform and the nominal options in the CMake build environment.

1.  Build LBANN from the local source repository and install the
    necessary dependencies into an environment.  The build script has
    a number of features that are described by the help message.
    Customization of the build is done via spack variants following
    the double dash.  An example of the invocation that installs
    dependencies and enables common variants is shown below:

    .. code-block:: bash

        <path to lbann repo>/scripts/build_lbann.sh -d -- +cuda +half

    Note that the named version and resulting environment can be
    controlled via the :code:`-l` flag. A full list of options can be
    viewed with the :code:`-h` flag.  External packages are setup via
    :code:`modules` and found via spack using the :code:`spack
    externals find` command.  If you want to provide your own modules
    just pass the :code:`--no-modules` flag to the
    :code:`build_lbann.sh` script to have it avoid loading what it
    thinks are good one.

    .. note:: Pro-tip: use a customized version label (via :code:`-l`)
              to create a build of LBANN tailored for your PR,
              experiment, etc.  Each version creates it's own module
              file, lives in its own Spack environment, and build
              directory.  The default version label is :code:`local`.

   .. warning:: Depending on the completeness of the externals
                specification, the initial build of all of the
                standard packages in Spack can take a long time.

   .. note:: The build script will automatically update your Spack
             repository to use clingo.  The manual instructions for
             doing this are detailed in the :ref:`user instructions
             <setting_up_clingo>`.  Note that if Spack's
             bootstrapping fails due to not finding a valid compiler,
             please refer to the explicit user instructions on how to
             have spack find a modern enough C++ compiler.

2.  Once the installation has completed, to run LBANN you will want to
    load the spack module for LBANN with one of the following
    commands, where the architecture and hash are printed at the end
    of the build script:

    .. code-block:: console

        module load lbann/local-<arch>-<hash>

    .. code-block:: console

        spack load lbann@local-<arch>-<hash>

3.  After the initial setup of the LBANN CMake environment, you can
    rebuild by activating the Spack environment and then re-running
    ninja.

    .. code-block:: console

        spack env activate -p lbann-<label>-<arch>
        spack build-env lbann -- bash
        cd <path to lbann repo>/spack-build-<hash>
        ninja install

For more control over the LBANN build, please see :ref:`the complete
documentation for building LBANN directly with CMake
<build-with-cmake>`.

--------------------------------------------
Debugging some common Spack related issues
--------------------------------------------

One common issue that can occur is that the modules can get out of
sync between what the LBANN environment does and the Spack defaults.
As a result the generated module files can get out of whack.  LBANN
uses a module hierarchy naming scheme that is compatible with other
modules, provides for name collision, and reduces the clutter in the
module name.  If your modules are not working you can regenerate them
in a LBANN Spack environment compatible approach:

    .. code-block:: console

       spack env activate -p lbann-<label>-<arch>
       spack module lmod refresh --delete-tree
       spack module tcl refresh --delete-tree

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using Python packages with LBANN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LBANN toolkit engages with Python and Python packages in two key
ways.  The first of which is that LBANN has a Python Front-End (PFE)
that use Python to create a model for the LBANN executable as well as
launch jobs.  The second aspect is that LBANN has a Python data reader
that can execute Python code to process data.  To use a common Python
installation with both LBANN as well as the any external Python
packages the user can either use Spack to build everything or to use
PIP to install Python-only requirements.

1. [Spack Option] If the user wants to ensure that all aspects of the
   LBANN + Python packages should work together they can be all added
   to the same Spack environment and compiled together.  The
   `build_lbann.sh` script can make this easier with the following
   command line options:

   .. code-block:: console

        <path to lbann repo>/scripts/build_lbann.sh <options>
                    -p py-scipy@1.8.1 --pip pytest --pip tqdm
                    -- <variants>

or

   .. code-block:: console

        <path to lbann repo>/scripts/build_lbann.sh <options>
                    -e applications/ATOM/external_packages_atom.sh
                    -- <variants>
or

   .. code-block:: console

        <path to lbann repo>/scripts/build_lbann.sh <options>
                    -p py-numpy
                    -- <variants>

2.  A second option is to install additional Python packages via Pip.
    To ensure that this is done with the correct version of Python
    that LBANN was built with, activate the spack environment:

   .. code-block:: console

        spack env activate -p <name of environment>
        <path to lbann repo>/scripts/build_lbann.sh <options>
                    --pip <requirements.txt>
                    -- <variants>

Regardless of the approach that is used, certain Python packages work
best when compiled with the same compiler (and C++ library) as LBANN,
specifically NumPy.  So by default the `build_lbann.sh` script will
install NumPy along with the LBANN in the same environment.

   .. warning:: Note that beyond using Spack for NumPy, mixing the
                installation of Python extra packages via Spack and
                Pip is definitely an expert-mode activity (YMMV).

------------------------------
Advanced build methods
------------------------------

If you want to build LBANN with local versions of Hydrogen,
DiHydrogen, or Aluminum, you can instruct the script and spack to
build from local repositories:

    .. code-block:: console

        <path to lbann repo>/scripts/build_lbann.sh
                    --hydrogen-repo <path>/Hydrogen.git
                    --aluminum-repo <path>/Aluminum.git
                    --dihydrogen-repo <path>/DiHydrogen.git

.. toctree::
   :maxdepth: 1

   build_osx
   build_with_cmake
   build_containers
   build_llnl_idiosyncracies
