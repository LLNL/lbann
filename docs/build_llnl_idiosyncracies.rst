Specific information for Livermore Computing (LC) systems
============================================================

.. warning:: Many features below make assumptions that users belong to
             certain groups on LC systems. Any information contained
             here should not be considered general-purpose and any
             examples are not expected to work except for certain
             users on LLNL's LC systems.

The :code:`build_lbann_lc.sh` script
----------------------------------------

The :code:`build_lbann_lc.sh` script in the :code:`scripts/` directory
is a historical script with logic to choose the "right" compilers
and grab all the LC-installed dependencies. It is updated on an
*ad-hoc* basis by the subset of developers who use it and it should
not be relied upon as a replacement for other methods described in
this guide.

.. warning:: Certain paths through this script require access to a
             certain linux group on the Livermore Computing
             machines (LC) at LLNL.


Pre-installed Binary Packages
------------------------------

The LC machines have many instances of cuDNN and NCCL installed in
locations shared by the :code:`brain` group. These may be consistently
detected by CMake by :code:`export`-ing their locations into the
shell:

.. code-block:: bash

    export CUDNN_DIR=/usr/WS2/brain/cudnn/cudnn-7.4.2/cuda-10.0_x86_64
    export NCCL_DIR=/usr/WS2/brain/nccl2/nccl_2.4.2-1+cuda10.0_x86_64

Notice that this is specific to using CUDA 10.0 on an x86_64 LC
machine. This is a shortcut around formally passing this location as a
cache variable to all relevant CMake projects. The cache method for
passing these to the LBANN CMake is:

.. code-block:: bash

    cmake CUDNN_DIR=/usr/WS2/brain/cudnn/cudnn-7.4.2/cuda-10.0_x86_64 \
        <other args> \
        /path/to/lbann

LBANN will detect NCCL automatically from the Aluminum import; there
should be no need to pass :code:`NCCL_DIR` to the LBANN CMake.

The Superbuild, on the other hand, may require both :code:`CUDNN_DIR`
and :code:`NCCL_DIR` if building both Aluminum and LBANN. Such an
invocation might be:

.. code-block:: bash

    cmake -DLBANN_SB_BUILD_ALUMINUM=ON -DALUMINUM_ENABLE_NCCL=ON \
        -DLBANN_SB_FWD_ALUMINUM_NCCL_DIR=/usr/WS2/brain/nccl2/nccl_2.4.2-1+cuda10.0_x86_64 \
        -DLBANN_SB_BUILD_HYDROGEN=ON \
        -DLBANN_SB_BUILD_LBANN=ON \
        -DLBANN_SB_FWD_CUDNN_DIR=/usr/WS2/brain/cudnn/cudnn-7.4.2/cuda-10.0_x86_64 \
        <other options> \
        /path/to/lbann/superbuild
