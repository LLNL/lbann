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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setup Spack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Download and install `Spack <https://github.com/llnl/spack>`_.
    Additionally setup shell support as discussed
    `here <https://spack.readthedocs.io/en/latest/module_file_support.html#id2>`_.

    .. code-block:: bash

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

.. _install_lbann_as_user:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building & Installing LBANN as a user
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Spack setup and installed into your path, it can be used to
install the LBANN executables. This approach is appropriate for users
that want to train new or existing models using the Python front-end.

.. note:: If your model requires custom layers or data readers, you
          may need to install LBANN as a developer, which would allow
          you to modify and recompile the source code.

Users comfortable with Spack and `its idioms for installing packages
<https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html>`_
or those who already have `customizations to their Spack ecosystem
<https://spack.readthedocs.io/en/latest/configuration.html>`_ in place
may simply use

.. code-block:: bash

   spack install lbann <customization options>

In this case, it is not even necessary to clone the LBANN repository
from Github; Spack will handle this in its installation.

For users that are new to spack, LBANN provides a script that will do
some basic configuration and then install LBANN using the Spack
environment method:

.. code-block:: bash

   <path lbann repo>/scripts/install_lbann.sh -e lbann
   spack env activate -p lbann

Options exist in the script to disable the GPUs and change the
name of the Spack environment. These can be viewed by passing the
:code:`-h` option to the script.

.. note:: Currently this script will clone a second LBANN repository
          that Spack will use to build the LBANN library and
          executables. We are working on simplifying this further.


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
Expert developers should refer to :ref:`the "Superbuild" documentation
<building-with-the-superbuild>` for a list and descriptions of all
CMake flags known to LBANN's "Superbuild" build system.

1.  Install all of the external packages via Spack (Aluminum,
    Hydrogen, etc).

    Install packages into a Spack environment. This is only done when
    initially installing or upgrading the dependencies. LBANN provides
    a script to install the basic dependencies in their default
    configurations and it can be found at:

    .. code-block:: bash

        <path to lbann repo>/scripts/install_lbann.sh -d

    Note that the named environment can be controlled via the
    :code:`-e` flag. A full list of options can be viewed with the
    :code:`-h` flag.

2.  Setup the LBANN CMake environment using the Spack environment for
    the dependencies.

    .. code-block:: bash

        <path to lbann repo>/scripts/build_lbann_from_source.sh


   Options exist in the script to disable the GPUs, set a build and
   install prefix, separately set the build and install
   directories, or use a different spack environment. These options
   can be viewed using the :code:`-h` flag.

   The environments provided by this script have a set of external
   packages and compilers that are installed on an LLNL LC CZ, NERSC,
   or LLNL-configured OS X system. If you are not on one of these
   systems, please update the externals and compilers for your system
   environment. Alternatively, you can create baseline versions of
   the user-level Spack configuration files and remove the externals
   and compilers from the :code:`spack.yaml` file. More details are
   provided :ref:`here <setup-spack-env>`.

   .. warning:: Depending on the completeness of the externals
                specification, the initial build of all of the
                standard packages in Spack can take a long time.

3.  Once the installation has completed, you can load the module file
    for LBANN with the following command

    .. code-block:: console

        ml use <path to installation>/etc/modulefiles
        ml load lbann-0.99.0


    For advanced users, :ref:`the LBANN superbuild system
    <building-with-the-superbuild>` provides additional control over
    the dependencies, especially Aluminum and Hydrogen.

4.  After the initial setup of the LBANN CMake environment, you can
    rebuild by activating the Spack environment and then re-running
    ninja.

    .. code-block:: console

         spack env activate -p <environmment>
         cd <build directory>/lbann/build
         unset CPATH # Can cause bad include resolution
         ninja

For more control over the LBANN build, please see :ref:`the complete
documentation for building LBANN directly with CMake
<build-with-cmake>`.

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
