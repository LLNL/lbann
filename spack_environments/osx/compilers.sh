#!/bin/sh

COMPILER_ALL_PACKAGES=$(cat <<EOF
      compiler: [clang@9.0.1 arch=darwin-mojave-skylake, clang@9.0.0 arch=darwin-mojave-skylake]
EOF
)

COMPILER_DEFINITIONS=$(cat <<EOF
  compilers:
  - compiler:
      environment: {}
      extra_rpaths: []
      flags: {}
      modules: []
      operating_system: mojave
      paths:
        cc: /usr/local/Cellar/llvm/9.0.1/bin/clang
        cxx: /usr/local/Cellar/llvm/9.0.1/bin/clang++
        f77: /usr/local/bin/gfortran
        fc: /usr/local/bin/gfortran
      spec: clang@9.0.1
      target: x86_64
  - compiler:
      environment: {}
      extra_rpaths: []
      flags: {}
      modules: []
      operating_system: mojave
      paths:
        cc: /usr/local/Cellar/llvm/9.0.0_1/bin/clang
        cxx: /usr/local/Cellar/llvm/9.0.0_1/bin/clang++
        f77: /usr/local/bin/gfortran
        fc: /usr/local/bin/gfortran
      spec: clang@9.0.0
      target: x86_64
EOF
)
