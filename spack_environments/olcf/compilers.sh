#!/bin/sh

COMPILER_ALL_PACKAGES=$(cat <<EOF
      compiler:
      - gcc@7.4.0 arch=linux-rhel7-power9le
      - gcc@8.1.1 arch=linux-rhel7-power9le
EOF
)

COMPILER_DEFINITIONS=$(cat <<EOF
  compilers:
  - compiler:
      environment: 
        unset: []
      extra_rpaths: []
      flags: {}
      modules: []
      operating_system: rhel7
      paths:
        cc: /sw/summit/gcc/7.4.0/bin/gcc
        cxx: /sw/summit/gcc/7.4.0/bin/g++
        f77: /sw/summit/gcc/7.4.0/bin/gfortran
        fc: /sw/summit/gcc/7.4.0/bin/gfortran
      spec: gcc@7.4.0
      target: ppc64le
  - compiler:
     environment:
       unset: []
     extra_rpaths: []
     flags: {}
     modules: []
     operating_system: rhel7
     paths:
       cc: /sw/summit/gcc/8.1.1/bin/gcc
       cxx: /sw/summit/gcc/8.1.1/bin/g++
       f77: /sw/summit/gcc/8.1.1/bin/gfortran
       fc: /sw/summit/gcc/8.1.1/bin/gfortran
     spec: gcc@8.1.1
     target: ppc64le
EOF
)
