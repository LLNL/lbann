#!/bin/sh

STD_PACKAGES=$(cat <<EOF
    cereal::
      buildable: true
      version: [1.2.2]

    conduit::
      buildable: true
      variants: ~doc~doxygen+hdf5~hdf5_compat+mpi+python+shared~silo
      version: [0.5.1]

    cnpy::
      buildable: true
      variants: build_type=RelWithDebInfo
      version: [master]

    cub::
      buildable: true
      version: [1.8.0]

    nccl::
      buildable: true
      version: [2.7.8-1]

    protobuf::
      buildable: True
      variants: build_type=Release +shared
      version: [3.10.0]

    py-numpy::
      buildable: True
      version: [1.16.2]

    py-protobuf::
      buildable: True
      variants: +cpp
      version: [3.10.0]

    zlib::
      buildable: True
      version: [1.2.11]
EOF
)

STD_MODULES=$(cat <<EOF
  modules:
    enable::
      - tcl
      - lmod
    lmod::
      hash_length: 3
      core_compilers:
        - 'gcc@7.3.0'
        - 'gcc@7.3.1'
      projections:
        all: '\${PACKAGE}/\${VERSION}-\${COMPILERNAME}-\${COMPILERVER}'
      blacklist:
        - '%gcc@4.8'
        - '%gcc@4.9.3'
      hierarchy:
        - 'mpi'
        - 'lapack'
      all:
        autoload: 'direct'
        suffixes:
          '^openblas': openblas
          '^netlib-lapack': netlib
        filter:
          # Exclude changes to any of these variables
          environment_blacklist: ['CPATH', 'LIBRARY_PATH']
      ^python:
        autoload:  'direct'
    tcl:
      hash_length: 3
      core_compilers:
        - 'gcc@7.3.0'
        - 'gcc@7.3.1'
      projections:
        all: '\${PACKAGE}/\${VERSION}-\${COMPILERNAME}-\${COMPILERVER}'
      whitelist:
        - gcc
      blacklist:
        - '%gcc@4.8'
        - '%gcc@4.9.3'
      all:
        autoload: 'direct'
        suffixes:
          '^openblas': openblas
          '^netlib-lapack': netlib
        filter:
          # Exclude changes to any of these variables
          environment_blacklist: ['CPATH', 'LIBRARY_PATH']
      ^python:
        autoload:  'direct'
EOF
)
