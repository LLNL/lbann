#!/bin/sh

COMPILER_ALL_PACKAGES=$(cat <<EOF
      compiler: [gcc@7.3.0 arch=linux-rhel7-broadwell, gcc@7.3.0 arch=linux-rhel7-haswell, gcc@7.3.1 arch=linux-rhel7-power9le, gcc@7.3.1 arch=linux-rhel7-power8le]
EOF
)

COMPILER_DEFINITIONS=$(cat <<EOF
  compilers:
  - compiler:
      environment: {}
      extra_rpaths: []
      flags: {}
      modules: []
      operating_system: rhel7
      paths:
        cc: /usr/tce/packages/gcc/gcc-7.3.1/bin/gcc
        cxx: /usr/tce/packages/gcc/gcc-7.3.1/bin/g++
        f77: /usr/tce/packages/gcc/gcc-7.3.1/bin/gfortran
        fc: /usr/tce/packages/gcc/gcc-7.3.1/bin/gfortran
      spec: gcc@7.3.1
      target: ppc64le
  - compiler:
      environment: {}
      extra_rpaths: []
      flags: {}
      modules: []
      operating_system: rhel7
      paths:
        cc: /usr/tce/packages/gcc/gcc-7.3.0/bin/gcc
        cxx: /usr/tce/packages/gcc/gcc-7.3.0/bin/g++
        f77: /usr/tce/packages/gcc/gcc-7.3.0/bin/gfortran
        fc: /usr/tce/packages/gcc/gcc-7.3.0/bin/gfortran
      spec: gcc@7.3.0
      target: x86_64
EOF
)
