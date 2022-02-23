.. role:: python(code)
          :language: python


.. _math-layers:

====================================
Math Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`DFTAbs`, "Absolute value of discrete Fourier transform"
   :ref:`MatMul`, "Matrix multiplication"

________________________________________


.. _DFTAbs:

----------------------------------------
DFTAbs
----------------------------------------

Absolute value of discrete Fourier transform. One-, two-, or
three-dimensional data is allowed. The implementation is meant to be
as flexible as possible. We use FFTW for the CPU implementation;
whichever types your implementation of FFTW supports will be supported
in this layer at runtime. The GPU implementation uses cuFFT on NVIDIA
GPUs and will support float and double at runtime (assuming CUDA
support is enabled). A future implementation will support rocFFT for
AMD GPUs.

Currently, LBANN only supports outputting the same type that is used
as input. As such, in forward propagation, this will do a DFT and then
compute the absolute value of the output implicitly. The intention is
to support immediate customer need now; we will generalize this as
LBANN learns to support different input/output data types.

Arguments: None

:ref:`Back to Top<math-layers>`

________________________________________


.. _MatMul:

----------------------------------------
MatMul
----------------------------------------

The MatMul layer performs Matrix multiplication.

Performs matrix product of two 2D input tensors. If the input tensors
are 3D, then matrix products are computed independently over the first
dimension, in a similar manner as NumPy's matmul function.

Arguments:

   :transpose_a: (``bool``) Whether to transpose matrices from first
                 input tensor

   :transpose_b: (``bool``) Whether to transpose matrices from second
                 input tensor

:ref:`Back to Top<math-layers>`
