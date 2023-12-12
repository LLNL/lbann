JIT-Compiled Layers
===================

LBANN implements a dynamic dispatcher for JIT-compiled distconv :ref:`Convolution` layers.
The dispatcher can be used to load fast implementations of convolutions on specific
shape and stride configurations (most notably distconv-enabled convolutions).
For implementing the JIT-compiled convolutions, we recommend using `DaCe <https://github.com/spcl/dace>`_
for API compatibility.

You can use the dynamic dispatcher by compiling LBANN with support for JIT-compiled convolutions
(adding the ``+dace`` variant with Spack, or ``-D LBANN_SB_FWD_DiHydrogen_H2_ENABLE_DACE=ON`` with CMake).
If support is enabled, you should see the following printout on initialization::

  DiHydrogen Features:
  DaCe : enabled



How it works
------------

When a convolution layer is called for the first time, the dynamic dispatcher looks for a shared library
(e.g., an ``.so`` or ``.dylib`` file) under the folder configured by the ``DISTCONV_JIT_CACHEPATH``
environment variable. Within the folder, LBANN will search for a file with the pattern
``lib<OPERATION DESCRIPTOR>.so``, where operation descriptor matches the specific configuration of
the convolution (i.e., dimensionality, shape, tensor strides). If the file is found, it will be used
instead of the standard vendor library implementation.

To find out what files are being queried, enable the verbose JIT-compiler mode by setting the
``DISTCONV_JIT_VERBOSE`` environment variable to ``1``.

Workflow
--------

Currently, the only way to work with LBANN and JIT-compiled convolutions is to run with verbose
mode enabled and find which files are queried. Then, run DaCe to generate a convolution implementation
(and tune it as necessary). See examples of convolution implementations at ``applications/physics/cosmology/cosmoflow/DaCe_kernels``.
Finally, place the shared libraries in the cache path and rerun LBANN.



Symbolic configuration
----------------------

In addition to being a concrete integer, the local mini-batch size of a JIT-compiled convolution
can be given as a symbolic variable. If compiled with symbolic mini-batch size, the corresponding
shape dimension should be given as ``B`` instead of a number.

LBANN will first search for a concrete local mini-batch size, and if not found will also look for
a dynamic mini-batch size.

Technical descriptor information
--------------------------------

The following is used as the convolution descriptor format (the bullet points are separated by
underscores):

  * ``conv<N>d`` specifies the convolution dimensionality. Up to 3-dimensional convolutions are currently supported.
  * A 5-dimensional shape of the input data tensor (``x`` or ``dx``), separated by underscores. The minibatch size can be symbolic (see above).
    For lower-dimensional tensors, add zeroes to the suffix (for example: ``32_3_224_224_0``).
  * A 5-dimensional stride (i.e., the number of elements to jump one value in the given dimension) of the input data tensor.
  * 5-dimensional shape of the weight tensor (``w``, ``dw``). The weight tensor is assumed to be contiguous (and the strides
    are up to the implementation's discretion).
  * 5-dimensional shape of the output data tensor (``y``, ``dy``).
  * 5-dimensional strides of the output data tensor.
  * Convolution parameters:
    * 3-dimensional (use zero suffix for fewer dimensions) padding
    * 3-dimensional convolution strides
    * 3-dimensional convolution dilations
    * Number of convolution groups for grouped convolution (use 1 for standard convolution)
  * ``{fwd,bwdfilt,bwddata}`` specifies the kind of the operation performed: Forward, Backpropagation (filter), or Backpropagation (data).


As an example, ``libconv3d_B_1_16_16_16_.so`` computes a 3-dimensional convolution on a 16x16x16 input with one channel, laid out as NCDHW, with a 3x3x3 kernel and 6 output channels.
