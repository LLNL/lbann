.. _building-with-the-superbuild:

=======================================================
Building an entire ecosystem with the "Superbuild"
=======================================================

.. warning:: This is primarily for developer convenience and is not
             meant to be robust to all possible use-cases for LBANN.

LBANN includes CMake :code:`ExternalProject` definitions for a large
portion of its dependency graph. The following dependencies are
supported. These are one or two edges from LBANN in the dependency
DAG.

+ Aluminum
+ CEREAL
+ CNPY
+ CONDUIT
+ `CUB <https://github.com/nvlabs/cub>`_. This is used by Hydrogen for
  efficiently managing GPU memory.
+ `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_. This is a
  dependency of CONDUIT.
+ Hydrogen
+ `JPEG-turbo <https://github.com/libjpeg-turbo/libjpeg-turbo>`_. This
  is a dependency of OpenCV.
+ `OpenBLAS <https://github.com/xianyi/OpenBLAS.git>`_. This is an
  optional dependency of Hydrogen. It is recommended if your system
  does not have a system-optimized BLAS distribution (e.g., Intel's MKL).
+ OpenCV
+ Protobuf

The following dependencies are known to exist but for some reason or
another are not supported by the superbuild framework.

+ cuDNN is a freely available binary package available from NVIDIA.
+ NCCL is a freely available binary package available from
  NVIDIA. Inspired users may also build it from source from its
  `github repository <https://github.com/nvidia/nccl>`_.
+ HWLOC is often installed by default, especially on large
  supercomputers. Certain components may require superuser access to
  configure, but these features are not used by LBANN. If it is not
  available, ask the system administrators, consult the package
  manager, install using Spack, or build from
  `source <https://www.open-mpi.org/projects/hwloc>`_.

The superbuild system is itself a CMake project rooted in
:code:`$LBANN_HOME/superbuild` (distinct from the LBANN CMake project rooted
in :code:`$LBANN_HOME`). Options that control the superbuild system are
prefixed with :code:`LBANN_SB_`; other options that appear in a CMake
invocation for the superbuild are either interpreted on a sub-project
basis or forwarded to certain sub-projects.

--------------------------------------------------
Choosing packages to build in the Superbuild
--------------------------------------------------

The superbuild system is *constructive* or *additive*; that is, it
will only build the packages that it is asked to build. Any required
package that is not requested is assumed to exist on the system by the
time it is needed by whichever package requires it. For example, if
HDF5 is provided by the system administrators on a system, it does not
need to be built and CONDUIT can be built by pointing its build to the
system HDF5.

Packages are included in a superbuild by passing
:code:`LBANN_SB_BUILD_<PKG>` options to CMake *for each package* that
it should build, including LBANN itself. E.g.,

.. code-block:: bash

    cmake \
      -DLBANN_SB_BUILD_ALUMINUM=ON \
      -DLBANN_SB_BUILD_HYDROGEN=ON \
      -DLBANN_SB_BUILD_LBANN=ON \
      /path/to/lbann/superbuild

will invoke the superbuild to build Aluminum, Hydrogen, and LBANN
*only*. Acceptable values for :code:`<PKG>` are :code:`ALUMINUM`,
:code:`CEREAL`, :code:`CNPY`, :code:`CONDUIT`, :code:`CUB`,
:code:`HDF5`, :code:`HYDROGEN`, :code:`JPEG_TURBO`, :code:`OPENCV`,
:code:`PROTOBUF` and :code:`LBANN`.


Forwarding options to sub-projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The subprojects are largely pre-configured to "do the right thing" for
building LBANN. However, there are some variables that users of the
superbuild system may need to control. These are exposed as regular
CMake options in the individual projects' CMakeLists and can be viewed
by running, e.g.,

.. code-block:: bash

    cmake -L superbuild/<pkg>/CMakeLists.txt

Several significant CMake flags are automatically forwarded from the
superbuild CMake to subprojects. These are generally "typical" CMake
flags (but not all; if something is missing, open please
`an issue <https://github.com/llnl/lbann/issues)>`_. Some examples are

+ :code:`CMAKE_INSTALL_PREFIX`
+ :code:`CMAKE_BUILD_TYPE`
+ :code:`CMAKE_<LANG>_COMPILER`
+ :code:`CMAKE_<LANG>_FLAGS`

To accommodate developers working on edge-cases with these
dependencies, any flag may be forwarded to any CMake-built package
using the following syntax:
:code:`LBANN_SB_FWD_<PKG>_<OPTION>=<VALUE>`. This will result in a cache
variable being sent to the CMake command for :code:`<PKG>` with the form

.. code-block:: bash

    -D<OPTION>=<VALUE>

The :code:`<OPTION>` may be something specific to :code:`<PKG>` or it
may be a CMake flag that is not automatically forwarded. For example,
the following CMake invocation would send
:code:`CMAKE_INTERPROCEDURAL_OPTIMIZATION` to the :code:`HYDROGEN`
package and :code:`SPHINX_DIR` to :code:`LBANN`:

.. code-block:: bash

    cmake -D LBANN_SB_BUILD_HYDROGEN=ON \
      -D LBANN_SB_BUILD_LBANN=ON \
      -D LBANN_SB_FWD_HYDROGEN_CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
      -D LBANN_SB_FWD_LBANN_SPHINX_DIR=/path/to/sphinx \
      /path/to/superbuild

-----------------------------------
Special targets in the Superbuild
-----------------------------------

Modern shells should be able to tab-complete the names of targets in
Makefiles or Ninja files, and IDEs should display all targets
interactively. The superbuild should create project-level targets for
all of the subprojects; these match the :code:`<PKG>` values noted
above. For example, after a successful CMake configuration of the
superbuild using the Ninja generator, the command

.. code-block:: bash

    ninja HYDROGEN

will build the sub-DAG ending with Hydrogen. If
:code:`LBANN_SB_BUILD_LBANN=ON`, `ninja LBANN` is equivalent to
:code:`ninja` since LBANN depends on all other targets built by the
superbuild.

When building on UNIX platforms, the "Unix Makefiles" and "Ninja"
generators will have special targets defined for debugging superbuild
issues. These targets are :code:`gather-build` and
:code:`gather-log`. These create tarballs of the build system files
and the execution logs generated for the superbuild or during the
superbuild build phase, respectively. The target :code:`gather-all`
depends on both of these targets and may be used to generate both
tarballs at once. The resulting tarballs are helpful to the build
system maintainers for debugging build issues if using the superbuild
system.

------------------------------
A full superbuild example
------------------------------

A full invocation to the superbuild that builds all dependencies might
look like the following. This example will use a CUDA-enabled build
with Aluminum and CONDUIT support using the currently-load GCC
toolset. It assumes that desired flags are stored in
:code:`<LANG>_FLAGS` in the environment.

.. code-block:: bash

    cmake -GNinja \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=${PWD}/install \
        -D CMAKE_C_COMPILER=$(which gcc) \
        -D CMAKE_C_FLAGS="${C_FLAGS}" \
        -D CMAKE_CXX_COMPILER=$(which g++) \
        -D CMAKE_CXX_FLAGS="${CXX_FLAGS}" \
        -D CMAKE_Fortran_COMPILER=$(which gfortran) \
        -D CMAKE_Fortran_FLAGS="${Fortran_FLAGS}" \
        -D CMAKE_CUDA_COMPILER=$(which nvcc) \
        -D CMAKE_CUDA_FLAGS="${CUDA_FLAGS}" \
        \
        -D LBANN_SB_BUILD_CEREAL=ON \
        -D LBANN_SB_BUILD_CNPY=ON \
        -D LBANN_SB_BUILD_CONDUIT=ON \
        -D LBANN_SB_BUILD_CUB=ON \
        -D LBANN_SB_BUILD_HDF5=ON \
        -D LBANN_SB_BUILD_JPEG_TURBO=ON \
        -D LBANN_SB_BUILD_OPENBLAS=ON \
        -D LBANN_SB_BUILD_OPENCV=ON \
        -D LBANN_SB_BUILD_PROTOBUF=ON \
        \
        -D LBANN_SB_BUILD_ALUMINUM=ON \
        -D ALUMINUM_ENABLE_MPI_CUDA=ON \
        -D ALUMINUM_ENABLE_NCCL=ON \
        \
        -D LBANN_SB_BUILD_HYDROGEN=ON \
        -D Hydrogen_ENABLE_CUDA=ON \
        -D Hydrogen_ENABLE_CUB=ON \
        -D Hydrogen_ENABLE_ALUMINUM=ON \
        \
        -D LBANN_SB_BUILD_LBANN=ON \
        -D LBANN_WITH_ALUMINUM=ON \
        -D LBANN_WITH_CONDUIT=ON \
        -D LBANN_WITH_CUDA=ON \
        -D LBANN_WITH_NVPROF=ON \
        -D LBANN_WITH_TBINF=ON \
        -D LBANN_WITH_TOPO_AWARE=ON \
        -D LBANN_SEQUENTIAL_INITIALIZATION=OFF \
        -D LBANN_WARNINGS_AS_ERRORS=OFF \
        \
        /path/to/superbuild

Please report any issues with the superbuild on `Github
<https://github.com/llnl/lbann/issues>`_, but note that they will be
evaluated on a case-by-case basis and may not be fixed in a timely
manner or at all if they do not affect the development team. To
repeat, the superbuild exists for developer convenience and is not
meant to supplant a legitimate package manager.
