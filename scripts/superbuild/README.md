# LBANN SuperBuild

Welcome to the LBANN SuperBuild project. Let's begin by establishing
some boundaries and expectations.

## What this is

This SuperBuild infrastructure was originally developed as part of the
[LBANN Project](https://github.com/LLNL/LBANN) as a way to build the
first-order third-party dependencies for LBANN. It existed for some
time hidden behind the infamous and storied `build_lbann_lc.sh` shell
script. However, it was deprecated when Spack was determined to be
mature and capable enough to build LBANN. It was officially removed
from LBANN when the development team at the time had embraced the
Spack build.

Under normal circumstances, the official position of the LBANN team
(myself included) is that users and developers should use the LBANN
build script (which uses Spack under the hood). If you're here, it's
likely because the build script just isn't working for you, and
hopefully this will get you started.

The SuperBuild can build all of the first-order dependencies of LBANN
(except Python, due to its ubiquity), as well as a select few of its
second-order dependencies that are either deemed to have significant
performance impact (JPEG-turbo, OpenBLAS) or subject to version
constraints that are unlikely to be fulfilled by system defaults
(HDF5). It can also build LBANN itself, either from scratch or from a
source directory that a user provides. As long as you have a
mostly-sane operating system, this should get you a working build of
LBANN.

Above all, bear in mind: this is a *build tool*. It builds stuff, and
that's it.

The primary build targets are developer workstations (Linux and macOS)
and Department of Energy clusters/supercomputers. It is regularly
tested on a variety of LLNL machines as well as Perlmutter at NERSC.

## What this is NOT

This is not a package manager. Use `pacman`, `dnf`, `apt`, etc. for
that. Or use Spack if you're really desperate.

In particular, I don't track versions all that closely, and I don't do
anything to "update" packages beyond what CMake gives me for free.

I don't do anything special with Python. If you use this to build
LBANN and you can't run something due to a missing Python dependency,
install it however you would normally install a Python package.

The SuperBuild will do very little to babysit your build. It will
happily let you metaphorically shoot yourself in the foot. It does
exactly what you tell it to do.

# Supported packages

The following packages are known by the SuperBuild framework:

- [Aluminum](https://github.com/LLNL/Aluminum) - High-performance
  communication library that provides a stream-aware interface and
  semantics.
- [Catch2](https://github.com/catchorg/catch2) - A unit-testing
  framework for C++ packages. (Mostly for developers; also used by
  H2 and Hydrogen, if enabled.)
- [cereal](https://github.com/uscilab/cereal) - C++ serialization
  library.
- [Clara](https://github.com/catchorg/clara) - Argument parser (yes,
  it *is* deprecated, and no, I don't care. It works. It could be
  replaced by cxxopts or something if that changes.)
- [CNPY](https://github.com/rogersce/cnpy) - Read and write NumPy data
  in C++.
- [Conduit](https://github.com/LLNL/Conduit) - Yet another
  serialization format. Apparently there aren't enough of these.
- [DiHydrogen](https://github.com/LLNL/DiHydrogen) - Distributed
  tensors and associated operations upon them. This is also the
  current location of the legacy "DistConv" library used by LBANN.
- [HDF5](https://github.com/HDFGroup/hdf5) - Hierarchical data format,
  but for SCIENCE!
- [Hydrogen](https://github.com/LLNL/Elemental) - GPU-aware matrix
  algebra library.
- [JPEG-TURBO](https://github.com/libjpeg-turbo/libjpeg-turbo) - JPEG
  but in turbo mode. Zoom zoom zoom.
- [NCCL](https://github.com/NVIDIA/nccl) - The NVIDIA Collective
  Communications Library.
- [OpenBLAS](https://github.com/xianyi/OpenBLAS.git) - BLAS library
  for when your vendor doesn't do a good job.
- [OpenCV](https://github.com/opencv/opencv) - Computer vision
  library.
- [protobuf](https://github.com/protocolbuffers/protobuf.git) - And
  yet *another* serialization format that LBANN (and others) (ab)use
  for model topology description and configuration.
- [RCCL](https://github.com/ROCm/rccl) - The ROCm Communication
  Collectives Library.
- [spdlog](https://github.com/gabime/spdlog) - Fast C++ logging
  library.
- [zstr](https://github.com/mateidavid/zstr) - C++ ZLib wrapper.

The framework is "opt-in". No package will be built unless the user
explicitly enables it when they configure the SuperBuild project.

# CMake configuration

The SuperBuild project is configured and run through
[CMake](https://cmake.org). I personally advocate using the
[Ninja](https://ninja-build.org/) generator, but they should
work. Under the hood, the system works by defining a series of [CMake
External
Projects](https://cmake.org/cmake/help/git-stage/module/ExternalProject.html)
that is controlled by special CMake arguments, documented below.

An important bit to note is that `ExternalProject` generates the
*infratructure* to configure, build and install other projects.
*However*, these projects are *NOT* configured when the SuperBuild is
generated. This implies that, for things like library- or package-
finding in these projects, the environment that matters is *not* the
environment that exists when `cmake` is run to configure the
SuperBuild but rather the environment that is exported to the
subsequent invocation of the generated build system (e.g., `ninja` or
`make` or whatever). This should be borne in mind when thinking about
how to ensure that packages are resolving dependencies appropriately.

On this note, the SuperBuild only offers dependency-resolution
assistance with packages built by the SuperBuild. For example, if a
user builds HDF5 *and* Conduit with *the same* SuperBuild, the
SuperBuild itself will attempt to forward the correct HDF5 directory
to Conduit so that Conduit's configuration finds the HDF5 that was
built by this SuperBuild build system. However, if some other HDF5 is
to be used, the onus is on the user to forward the necessary
information about its existence and location to the Conduit package in
the SuperBuild (e.g., by setting
`LBANN_SB_FWD_Conduit_HDF5_DIR=/path/to/hdf5/prefix` in the SuperBuild
configuration line, [as described
below](#passing-options-to-superbuild-packages).

The main SuperBuild project is designed to be a passive task
manager. In CMake-speak, the language is `NONE` and it never calls a
`find_{path,package,executable,library}` function. It simply
configures a system of `ExternalProject` calls that do ALL of the work
of building these packages. This has the benefit that it's nearly
impossible for the configuration of the SuperBuild project itself to
fail. The downside is that it pushes all configuration issues to the
individual packages being built.

## SuperBuild CMake Options

There are several options and variables that can be used to control
the SuperBuild project itself. All arguments used by the SuperBuild
package are either standard CMake options (prefixed with `CMAKE_`) or
use the `LBANN_SB_` prefix.

- `LBANN_SB_BUILD_<PKG>` enables the build for `<PKG>`, which must
  match a package (case-sensitive) in the list of packages above.

- `LBANN_SB_CLONE_VIA_SSH` switches HTTPS GitHub URLs for SSH-based
  ones. The onus is on the user to ensure their SSH authentication
  method is setup and working.

- `LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY` defines how to modify the
  top-level `CMAKE_INSTALL_PREFIX` at the package level. Acceptible
  values are:
  - `COMMON` (default): Any package that doesn't specify its own
    prefix will install to the SuperBuild's `CMAKE_INSTALL_PREFIX` (if
    set).
  - `PKG`: Any package that doesn't specify its own prefix will
    install to `CMAKE_INSTALL_PREFIX/<PKG>`, with the casing of `PKG`
    matching that in the list of packages above.
  - `PKG_LC`: Any package that doesn't specify its own prefix will
    install to `CMAKE_INSTALL_PREFIX/<pkg>`, where `<pkg>` is the
    lower-cased version of the package name in the list of packages
    above.

The following variables will be forwarded to all GPU-enabled packages
if it can be detected that they are being built with GPU support. CUDA
packages will inherit

- `CMAKE_CUDA_ARCHITECTURES`

ROCm packages will inherit all of

- `CMAKE_HIP_ARCHITECTURES`
- `AMDGPU_TARGETS`
- `GPU_TARGETS`

If any of those are set, all will inherit that value and be forwarded
appropriately. (I have no explanation for why these are ALL needed,
but I get sporadic and nonsensical failures if I omit any. It is
zero-cost-to-me to forward them all, and therefore zero-value-to-me to
diagnose this (non-)issue.)

## Passing options to SuperBuild packages

The SuperBuild project attempts to expose mechanisms to allow even
fine-grained control over the build of its component packages. The
level of control is much higher if the package uses a CMake build, but
an effort is made for other types of projects. Packages have some
common options that are used to configure the `ExternalProject`
command in CMake. These are prefixed with `LBANN_SB_<PKG>_`, where
`<PKG>` matches a package (case-sensitive) in the list of packages
above. The mechanism for package-specific options are described in
detail [below](#options-specific-to-a-package).

### Common options

The following options are available to any package in the SuperBuild
system and may be used to customize the properties of the build of
those packages. It is the user's responsibility to ensure no
compatibility issues arise (e.g., using two different and incompatible
compilers to build two different packages). The options that are tied
to CMake options work by forwarding values to the
`ExternalProject`. This is (usually) easy for CMake projects, and a
"best effort" is made to map them to non-CMake-based build systems.

- `LBANN_SB_<PKG>_PREFIX` - The installation prefix for the `<PKG>`
  package. If not provided, this defaults to `CMAKE_INSTALL_PREFIX`
  modified according to `LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY`.
- `LBANN_SB_<PKG>_<LANG>_COMPILER` - The compiler for the `<LANG>`
  language for the `<PKG>` package. The default is the top-level
  `CMAKE_<LANG>_COMPILER`, if provided. Otherwise no value is
  forwarded and it is left to the package configure stage to deduce.
- `LBANN_SB_<PKG>_<LANG>_FLAGS` - The flags to pass to the compiler
  for the `<LANG>` language for the `<PKG>` package. The default is
  the top-level `CMAKE_<LANG>_FLAGS`, if provided. Otherwise no value
  is forwarded and it is left to the package configure stage to
  deduce.
- `LBANN_SB_<PKG>_SOURCE_DIR` - Specify a local directory to use to
  build the `<PKG>` package. If this is set, it must be the top-level
  directory for the build system of `<PKG>` (for a CMake package, the
  directory containing the top-level `CMakeLists.txt`). Using this
  option will disable ALL automatic Git steps and the source code will
  be built as-is.
- `LBANN_SB_<PKG>_URL` - The URL from which to clone the `<PKG>`
  package. This must be a Git repository. This defaults to some flavor
  of the "official" project repository (with a preference toward
  Github if multiple candidates exist). This is ignored if
  `LBANN_SB_<PKG>_SOURCE_DIR` is set.
- `LBANN_SB_<PKG>_TAG` - The Git tag to check out for the `<PKG>`
  package. The SuperBuild specifies a default tag for each package
  that is a version acceptible to use with the `develop` branch of
  LBANN.
- `LBANN_SB_<PKG>_CMAKE_GENERATOR` - Setup the CMake generator to use
  for the `<PKG>` package. This has no effect if the package is not a
  CMake package. It will fail at *SuperBuild build time* (during the
  configure step for `<PKG>`) if this is not a valid CMake generator
  string. The default value is the generator used to configure the
  SuperBuild.
- `LBANN_SB_<PKG>_BUILD_SHARED_LIBS` - Build shared libraries for the
  `<PKG>` package. This may not have an effect if the package is not a
  CMake package. The default value is the value of the top-level
  `BUILD_SHARED_LIBS`.
- `LBANN_SB_<PKG>_IPO` - Attempt to enable interprocedural
  optimization for the `<PKG>` package. The default value is the value
  of the top-level `CMAKE_INTERPROCEDURAL_OPTIMIZATION`. This only has
  an effect in CMake packages. Non-CMake packages should use compiler
  flags to manually specify IPO options.
- `LBANN_SB_<PKG>_PIC` - Attempt to enable
  position-independent code for the `<PKG>` package. This may override
  any inherited behavior from `LBANN_SB_<PKG>_BUILD_SHARED_LIBS`. This
  only has an effect in CMake packages. Non-CMake packages should use
  compiler flags to manually specify PIC/PIE flags.

### Options specific to a package

Packages in the SuperBuild that use CMake may have other options
forwarded to them by using a special variable syntax:
`LBANN_SB_FWD_<PKG>_<VARIABLE_NAME>=value`. For a project named `Foo`,
we could set a variable, `WITH_SOME_FEATURE=ON`, in the `Foo` build
system by passing `LBANN_SB_FWD_Foo_WITH_SOME_FEATURE=ON` to the
SuperBuild configuration. When the CMake `ExternalProject` is
generated for `Foo`, the configuration step will include `cmake
... -DWITH_SOME_FEATURE=ON ... /path/to/Foo/src` automatically.

Packages that do NOT use CMake builds (notably OpenBLAS) may have
additional options associated with them if they are coded into that
package's `CMakeLists.txt` format. At this time, there's no "hard and
fast rule" for the format of these options, and they may not follow
the `LBANN_SB_FWD_` syntax. Indeed, the option may not be forwarding
anything but rather manipulating the `ExternalProject_Add`
specification, so `_FWD_` may be semantically inappropriate. I haven't
decided yet if I want to push for it anyway for "consistency". In the
meantime, users are encouraged to consult the corresponding
`CMakeLists.txt` file or their CMake cache (`ccmake`, e.g.) for a list
of exposed options for these packages.

Some packages in the SuperBuild will have default settings that are
configured to match the "usual usage" for LBANN. These are exposed by
`CACHE` variables or explicit `option()` calls in the SuperBuild CMake
configuration, and users will see these appear in their
`CMakeCache.txt` even if they haven't explicitly set these
variables. They generally follow the same variable name syntax as
described above for consistency, though users are encouraged to verify
this in the appropriate `CMakeLists.txt` files.

# Debugging SuperBuild issues

SuperBuild packages are all built with a common structure. Unless
`LBANN_SB_<PKG>_SOURCE_DIR` was specified, the source clone will be
located in `<build_root>/<pkg>/src` The scripts that the SuperBuild
build stage will execute, as well as the log files generated by these
commands, are located in `<build_root>/<pkg>/stamp`. Auxiliary scripts
may be located in `<build_root>/<pkg>/tmp`. All CMake-based packages
have their build directories located in `<build_root>/<pkg>/build`.
Packages that build in the source tree (e.g., OpenBLAS) will not have
such a directory.

If configuration was successful, a fully-functional build system should
be present in the `<build_root>/<pkg>/build` directory, and a user
should be able to invoke `cmake --build <build_root>/<pkg>/build` to
perform a standalone (re)build. This can be useful for incremental
rebuilds while debugging and/or editing the package.

(TODO: Finish this section.)

# Getting help

If something fails at SuperBuild configuration time (i.e., when you
run `cmake` for this project), please [open an
issue](https://github.com/llnl/lbann/issues/new) and begin the title
with "[SuperBuild]" so I know to look. Please attach *THE WHOLE
OUTPUT* of CMake. I may also ask you for the log files it notes at the
end, usually `<BUILD_DIR>/CMakeFiles/CMake{Output,Error.log}`, so it's
not a bad idea to send those along, too.

If something fails in one of the package builds, and you're using the
Ninja or Makefiles generators, there's a target called `gather-all`
that will attempt to collect the relevant build system files and log
files for every package that was enabled. I will probably ask for the
tarballs generated by this target. When filing an issue, please
indicate which package is failing and if you have uncovered any
insight into why that's happening.
