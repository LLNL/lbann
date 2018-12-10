#!/bin/sh

# Set cluster-specific defaults
CLUSTER=${CLUSTER:-$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')}
case ${CLUSTER} in
    "catalyst")
        SCHEDULER=slurm
        PARTITION=${PARTITION:-pbatch}
        ACCOUNT=${ACCOUNT:-brain}
        CACHE_DIR=${CACHE_DIR:-/l/ssd}
        CORES_PER_NODE=24
        HAS_GPU=NO
        ;;
    "flash")
        SCHEDULER=slurm
        PARTITION=${PARTITION:-pbatch}
        CACHE_DIR=${CACHE_DIR:-/l/ssd}
        CORES_PER_NODE=20
        HAS_GPU=NO
        ;;
    "quartz")
        SCHEDULER=slurm
        PARTITION=${PARTITION:-pbatch}
        ACCOUNT=${ACCOUNT:-brain}
        CACHE_DIR=${CACHE_DIR:-/tmp/${USER}}
        CORES_PER_NODE=36
        HAS_GPU=NO
        ;;
    "surface")
        SCHEDULER=slurm
        PARTITION=${PARTITION:-pbatch}
        ACCOUNT=${ACCOUNT:-hpclearn}
        CACHE_DIR=${CACHE_DIR:-/tmp/${USER}}
        CORES_PER_NODE=16
        HAS_GPU=YES
        case ${PARTITION} in
            "pbatch")
                GPUS_PER_NODE=2
                ;;
            "gpgpu")
                GPUS_PER_NODE=4
                ;;
        esac
        ;;
    "ray")
        SCHEDULER=lsf
        PARTITION=${PARTITION:-pbatch}
        ACCOUNT=${ACCOUNT:-guests}
        CACHE_DIR=${CACHE_DIR:-/tmp}
        CORES_PER_NODE=20
        HAS_GPU=YES
        GPUS_PER_NODE=4
        ;;
    "pascal")
        SCHEDULER=slurm
        PARTITION=${PARTITION:-pbatch}
        ACCOUNT=${ACCOUNT:-hpcdl}
        CACHE_DIR=${CACHE_DIR:-/tmp/${USER}}
        CORES_PER_NODE=36
        HAS_GPU=YES
        GPUS_PER_NODE=2
        ;;
    *)
        SCHEDULER=slurm
        PARTITION=${PARTITION:-pbatch}
        ACCOUNT=${ACCOUNT:-brain}
        CACHE_DIR=${CACHE_DIR:-/tmp/${USER}}
        CORES_PER_NODE=1
        HAS_GPU=NO
        echo "Error: unrecognized system (${CLUSTER})"
        exit 1
        ;;
esac
