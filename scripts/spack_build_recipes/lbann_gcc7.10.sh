#!/bin/sh

CMD='spack setup lbann@local +el_cublas %gcc@7.1.0 ^elemental@master ^mvapich2@2.2'
echo $CMD
echo $CMD > build_lbann.sh
chmod +x build_lbann.sh
$CMD
