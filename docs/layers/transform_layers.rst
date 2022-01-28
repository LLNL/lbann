.. role:: python(code)
          :language: python


.. _transform-layers:

====================================
Transform Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`Reshape`, "Reinterpret tensor with new dimensions"
   :ref:`Pooling`, "Traverses the spatial dimensions of a data tensor with a
   sliding window and applies a reduction operation"
   :ref:`Concatenation`, "Concatenate tensors along specified dimension"
   :ref:`Slice`, "Slice tensor along specified dimension"
   :ref:`Split`, "Output the input tensor to multiple child layers"
   :ref:`Sum`, "Add multiple tensors"
   :ref:`Cross_Grid_Sum_Slice`, "Add tensors over multiple sub-grids and slice"
   :ref:`Cross_Grid_Sum`, "Add tensors over multiple sub-grids"
   :ref:`WeightedSum`, "Add tensors with scaling factors"
   :ref:`Unpooling`, "Transpose of pooling layer"
   :ref:`Hadamard`, "Entry-wise tensor product"
   :ref:`Constant`, "Output tensor filled with a single value"
   :ref:`Reduction`, "Reduce tensor to scalar"
   :ref:`Evaluation`, "Interface with objective function and metrics"
   :ref:`Gaussian`, "Random tensor with Gaussian/normal distribution"
   :ref:`Bernoulli`, "Random tensor with Bernoulli distribution"
   :ref:`Uniform`, "Random tensor with uniform distribution"
   :ref:`Crop`, "Extract crop from tensor at a position"
   :ref:`Dummy`, "Placeholder layer with no child layers"
   :ref:`StopGradient`, "Block error signals during back propagation"
   :ref:`InTopK`, "One-hot vector indicating top-k entries"
   :ref:`Sort`, "Sort tensor entries"
   :ref:`WeightsLayer`, "Output values from a weights tensor"
   :ref:`Tessellate`, "Repeat a tensor until it matches specified dimensions"
   :ref:`Scatter`, "Scatter values to specified tensor indices"
   :ref:`Gather`, "Gather values from specified tensor indices"
   :ref:`BatchwiseReduceSum`, "Sum of tensor entries over batch dimension"

**Deprecated transform layers**

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   CategoricalRandom, "Deprecated"
   DiscreteRandom, "Deprecated"

.. raw:: html

   <hr>

.. _Reshape:

----------------------------------------
Reshape
----------------------------------------

The Reshape layer reinterprets a tensor with new dimensions.

The input and output tensors must have the same number of
entries. This layer is very cheap since it just involves setting up
tensor views.


Arguments:

  :string dims:

     * Tensor dimensions
     * Space-separated list of integers. A single dimension may be
       -1, in which case the dimension is inferred.

Deprecated and unused

* int64 num_dims

.. raw:: html

   <hr>

.. _Pooling:

----------------------------------------
Pooling
----------------------------------------

The Pooling layer traverses the spatial dimensions of a data tensor
with a sliding window and applies a reduction operation.

Arguments:

  :string pool_mode:

     * (``string``, optional)
     * Pooling operation
     * Options: max, average, average_no_pad

  :int64 num_dims:

     * Number of spatial dimensions
     * The first data dimension is treated as the channel dimension,
       and all others are treated as spatial dimensions (recall that
       the mini-batch dimension is implicit).


  :bool has_vectors:

     * Whether to use vector-valued options
     * If true, then the pooling is configured with @c pool_dims, @c
       pool_pads, @c pool_strides. Otherwise, @c pool_dims_i, @c
       pool_pads_i, @c pool_strides_i.

  :string pool_dims:

    * Pooling window dimensions (vector-valued)
    * Space-separated list of integers, one for each spatial
      dimension. Used when @c has_vectors is enabled.

  :l_pads:

     * Pooling padding (vector-valued)
     * Space-separated list of integers, one for each spatial
       dimension. Used when @c has_vectors is enabled.

  :string pool_strides:

     * Pooling strides (vector-valued)
     * Space-separated list of integers, one for each spatial
       dimension. Used when @c has_vectors is enabled.

  :int64 pool_dims_i:

     * Pooling window dimension (integer-valued)
     * Used when @c has_vectors is disabled.

  :int64 pool_pads_i:

     * Pooling padding (integer-valued)
     * Used when @c has_vectors is disabled.

  :int64 pool_strides_i:

     * Pooling stride (integer-valued)
     * Used when @c has_vectors is disabled.


.. raw:: html

   <hr>

.. _:

----------------------------------------
Unpooling
----------------------------------------



.. raw:: html

   <hr>

.. _:

----------------------------------------
Slice
----------------------------------------
