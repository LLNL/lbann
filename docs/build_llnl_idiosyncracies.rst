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

