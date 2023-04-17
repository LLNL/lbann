.. role:: python(code)
          :language: python


.. _transform-layers:

====================================
 Transform Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`BatchwiseReduceSum`, "Sum of tensor entries over batch dimension"
   :ref:`Bernoulli`, "Random tensor with Bernoulli distribution"
   :ref:`Concatenation`, "Concatenate tensors along specified
   dimension"
   :ref:`Constant`, "Output tensor filled with a single value"
   :ref:`Crop`, "Extract crop from tensor at a position"
   :ref:`Cross_Grid_Sum`, "Add tensors over multiple sub-grids"
   :ref:`Cross_Grid_Sum_Slice`, "Add tensors over multiple sub-grids
   and slice"
   :ref:`Dummy`, "Placeholder layer with no child layers"
   :ref:`Evaluation`, "Interface with objective function and metrics"
   :ref:`Gather`, "Gather values from specified tensor indices"
   :ref:`Gaussian`, "Random tensor with Gaussian/normal distribution"
   :ref:`Hadamard`, "Entry-wise tensor product"
   :ref:`IdentityZero`, "Identity/zero function if layer is unfrozen/frozen."
   :ref:`InTopK`, "One-hot vector indicating top-k entries"
   :ref:`Pooling`, "Traverses the spatial dimensions of a data tensor
   with a sliding window and applies a reduction operation"
   :ref:`Reduction`, "Reduce tensor to scalar"
   :ref:`Reshape`, "Reinterpret tensor with new dimensions"
   :ref:`Scatter`, "Scatter values to specified tensor indices"
   :ref:`Slice`, "Slice tensor along specified dimension"
   :ref:`Sort`, "Sort tensor entries"
   :ref:`Split`, "Output the input tensor to multiple child layers"
   :ref:`StopGradient`, "Block error signals during back propagation"
   :ref:`Sum`, "Add multiple tensors"
   :ref:`TensorPermute`, "Permute the indices of a tensor"
   :ref:`Tessellate`, "Repeat a tensor until it matches specified
   dimensions"
   :ref:`Uniform`, "Random tensor with uniform distribution"
   :ref:`Unpooling`, "Transpose of pooling layer"
   :ref:`WeightedSum`, "Add tensors with scaling factors"
   :ref:`WeightsLayer`, "Output values from a weights tensor"


Deprecated transform layers:

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`CategoricalRandom`, "Deprecated"
   :ref:`DiscreteRandom`, "Deprecated"


________________________________________

.. _BatchwiseReduceSum:

----------------------------------------
BatchwiseReduceSum
----------------------------------------

The BatchwiseReduceSum layer is the sum of tensor entries over batch
dimension. The output tensor has same shape as input tensor.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _IdentityZero:

----------------------------------------
IdentityZero
----------------------------------------

The :python:`IdentityZero` layer is an output tensor filled with
either zeros or ones depending on if the layer is frozen or not. This
is useful for more complex training setups like GANs, where you want
to reuse the computational graph but switch loss functions.

Arguments:

   :num_neurons:

      (``string``) Tensor dimensions

      List of integers

:ref:`Back to Top<transform-layers>`

________________________________________

.. _Bernoulli:

----------------------------------------
Bernoulli
----------------------------------------

The :python:`Bernoulli` layer is a random tensor with a Bernoulli
distribution. Randomness is only applied during training. The tensor
is filled with zeros during evaluation.

Arguments:

   :prob: (``double``) Bernoulli distribution probability

   :neuron_dims:

      (``string``) Tensor dimensions

      List of integers

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Concatenation:

----------------------------------------
Concatenation
----------------------------------------

The :python:`Concatenation` layer concatenates tensors along specified
dimensions. All input tensors must have identical dimensions, except
for the concatenation dimension.

Arguments:

   :axis: (``int64``) Tensor dimension to concatenate along

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Constant:

----------------------------------------
Constant
----------------------------------------

The :python:`Constant` layer is an output tensor filled with a single
value.

Arguments:

   :value: (``double``) Value of tensor entries

   :num_neurons:

      (``string``) Tensor dimensions

      List of integers

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Crop:

----------------------------------------
Crop
----------------------------------------

The :python:`Crop` layer extracts a crop from a tensor at a
position. It expects two input tensors: an :math:`N` -D data tensor
and a 1D position vector with :math:`N` entries. The position vector
should be normalized so that values are in :math:`[0,1]` . For images
in CHW format, a position of (0,0,0) corresponds to the red-top-left
corner and (1,1,1) to the blue-bottom-right corner.

Arguments:

 :dims:

    (``string``) Crop dimensions
    List of integers

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Cross_Grid_Sum:

----------------------------------------
Cross_Grid_Sum
----------------------------------------

The :python:`Cross_Grid_Sum` layer adds tensors over multiple
sub-grids. This is experimental functionality for use with sub-grid
parallelism.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Cross_Grid_Sum_Slice:

----------------------------------------
Cross_Grid_Sum_Slice
----------------------------------------

The :python:`Cross_Grid_Sum_Slice` layer adds tensors over multiple
sub-grids and slices. This is experimental functionality for use with
sub-grid parallelism.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Dummy:

----------------------------------------
Dummy
----------------------------------------

The :python:`Dummy` layer is a placeholder layer with no child
layers. Rarely needed by users. This layer is used internally to
handle cases where a layer has no child layers.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Evaluation:

----------------------------------------
Evaluation
----------------------------------------

The :python:`Evaluation` layer is an interface with objective function
and metrics. Rarely needed by users. Evaluation layers are
automatically created when needed in the compute graph.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Gather:

----------------------------------------
Gather
----------------------------------------

The :python:`Gather` layer gathers values from specified tensor
indices. Expects two input tensors: an :math:`N` -D data tensor and a
1D index vector. For 1D data:

.. math::

   y[i] = x[\text{ind}[i]]

If an index is out-of-range, the corresponding output is set to zero.

For higher-dimensional data, the layer performs a gather along one
dimension. For example, with 2D data and axis=1,

.. math::

   y[i,j] = x[i,\text{ind}[j]]

Currently, only 1D and 2D data is supported.

The size of the the output tensor along the gather dimension is equal
to the size of the index vector. The remaining dimensions of the
output tensor are identical to the data tensor.

.. todo::
   Support higher-dimensional data

Arguments:

   :axis: (``google.protobuf.UInt64Value``) Dimensions to gather along

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Gaussian:

----------------------------------------
Gaussian
----------------------------------------

The :python:`Gaussian` layer is a random tensor with Gaussian/normal
distribution.

Arguments:

   :mean: (``double``) Distribution mean

   :stdev: (``double``) Distribution standard deviation

   :neuron_dims:

      (``string``) Tensor dimensions

      List of integers

   :training_only:

      (``bool``) Only generate random values during training

      If true, the tensor is filled with the distribution mean during
      evaluation.

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Hadamard:

----------------------------------------
Hadamard
----------------------------------------

The :python:`Hadamard` layer is an entry-wise tensor product.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _InTopK:

----------------------------------------
InTopK
----------------------------------------

The :python:`InTopK` layer is a one-hot vector indicating top-k
entries. Output tensor has same dimensions as input tensor. Output
entries corresponding to the top-k input entries are set to one and
the rest to zero. Ties are broken in favor of entries with smaller
indices.

Arguments:

   :k: (``int64``) Number of non-zeros in one-hot vector

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Pooling:

----------------------------------------
Pooling
----------------------------------------

The :python:`Pooling` layer traverses the spatial dimensions of a data
tensor with a sliding window and applies a reduction operation.

Arguments:

   :pool_mode:

      (``string``, optional) Pooling operation

      Options: max, average, average_no_pad

   :num_dims:

      (``int64``) Number of spatial dimensions

      The first data dimension is treated as the channel dimension,
      and all others are treated as spatial dimensions (recall that
      the mini-batch dimension is implicit).

   :has_vectors:

      (``bool``) Whether to use vector-valued options

      If true, then the pooling is configured with ``pool_dims``,
      ``pool_pads``, ``pool_strides``. Otherwise, ``pool_dims_i``,
      ``pool_pads_i``, ``pool_strides_i``.

   :pool_dims:

      (``string``) Pooling window dimensions (vector-valued)

      List of integers, one for each spatial
      dimension. Used when ``has_vectors`` is enabled.

   :pool_pads:

      (``string``) Pooling padding (vector-valued)

      List of integers, one for each spatial
      dimension. Used when ``has_vectors`` is enabled.

   :pool_strides:

      (``string``) Pooling strides (vector-valued)

      List of integers, one for each spatial
      dimension. Used when ``has_vectors`` is enabled.

   :pool_dims_i:

      (``int64``) Pooling window dimension (integer-valued)

      Used when ``has_vectors`` is disabled.

   :pool_pads_i:

      (``int64``) Pooling padding (integer-valued)

      Used when ``has_vectors`` is disabled.

   :pool_strides_i:

      (``int64``) Pooling stride (integer-valued)

      Used when ``has_vectors`` is disabled.

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Reduction:

----------------------------------------
Reduction
----------------------------------------

The :python:`Reduction` layer reduces a tensor to a scalar.

Arguments:

   :mode:

      (``string``, optional) Reduction operation

      Options: sum (default) or mean

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Reshape:

----------------------------------------
Reshape
----------------------------------------

The :python:`Reshape` layer reinterprets a tensor with new dimensions.

The input and output tensors must have the same number of
entries. This layer is very cheap since it just involves setting up
tensor views.

Arguments:

   :dims:

      (``string``) Tensor dimensions

      List of integers. A single dimension may be
      -1, in which case the dimension is inferred.

Deprecated and unused arguments:

   :num_dims: (``int64``)

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Scatter:

----------------------------------------
Scatter
----------------------------------------

The :python:`Scatter` layer scatters values to specified tensor
indices. Expects two input tensors: an :math:`N` -D data tensor and a
1D index vector. For 1D data:

.. math::

   y[\text{ind}[i]] = x[i]

Out-of-range indices are ignored.

For higher-dimensional data, the layer performs a scatter along one
dimension. For example, with 2D data and axis=1,

.. math::

   y[i,\text{ind}[j]] = x[i,j]


Currently, only 1D and 2D data is supported.

The size of the index vector must match the size of the data tensor
along the scatter dimension.

.. todo::
   Support higher-dimensional data

Arguments:

   :dims:

      (``string``) Output tensor dimensions

      List of integers. Number of dimensions must
      match data tensor.

   :axis: (``google.protobuf.UInt64Value``) Dimension to scatter along

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Slice:

----------------------------------------
Slice
----------------------------------------

The :python:`Slice` layer slices a tensor along a specified
dimension. The tensor is split along one dimension at user-specified
points, and each child layer recieves one piece.

Arguments:

   :axis: (``int64``) Tensor dimension to slice along

   :slice_points:

      (``string``) Positions at which to slice tensor

      List of integers. Slice points must be in
      ascending order and the number of slice points must be one
      greater than the number of child layers.

Deprecated arguments:

   :get_slice_points_from_reader: (``string``) Do not use unless using
                                  the Jag dataset.

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Sort:

----------------------------------------
Sort
----------------------------------------

The :python:`Sort` layer sorts tensor entries.

Arguments:

   :descending: (``bool``) Sort entries in descending order

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Split:

----------------------------------------
Split
----------------------------------------

The :python:`Split` layer outputs the input tensor to multiple child
layers.

Rarely needed by users. This layer is used internally to handle cases
where a layer outputs the same tensor to multiple child layers. From a
usage perspective, there is little difference from an identity layer.

This is not to be confused with the split operation in NumPy, PyTorch
or TensorFlow. The name refers to splits in the compute graph.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _StopGradient:

----------------------------------------
StopGradient
----------------------------------------

The :python:`StopGradient` layer blocks error signals during back
propagation.

The output is identical to the input, but the back propagation output
(i.e. the error signal) is always zero. Compare with the stop_gradient
operation in TensorFlow and Keras. Note that this means that computed
gradients in preceeding layers are not exact gradients of the
objective function.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Sum:

----------------------------------------
Sum
----------------------------------------

The :python:`Sum` layer calculates the element-wise sum of each of the
input tensors.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _TensorPermute:

----------------------------------------
TensorPermute
----------------------------------------

The :python:`TensorPermute` layer permutes the indices of a tensor, similar
to a transposition.

It expects one input tensor of order N, and a length N array of
permuted indices [0..N-1], with respect to the input tensor
dimensions. Therefore, passing ``axes=[0,1,2]`` for a rank-3 tensor
will invoke a copy.

At this time, only permutations are supported. Each
index must be accounted for in the permuted array.

Arguments:

   :axes:

      (``uint32``) Permuted tensor dimensions 

      List of integers

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Tessellate:

----------------------------------------
Tessellate
----------------------------------------

The :python:`Tessallate` layer repeats a tensor until it matches
specified dimensions.

The output tensor dimensions do not need to be integer multiples of
the input dimensions. Compare with the NumPy ``tile`` function.

As an example, tessellating a :math:`2 \times 2` matrix into a
:math:`3 \times 4` matrix looks like the following:

.. math::

   \begin{bmatrix}
     1 & 2 \\
     3 & 4
   \end{bmatrix}
   \rightarrow
   \begin{bmatrix}
     1 & 2 & 1 & 2 \\
     3 & 4 & 3 & 4 \\
     1 & 2 & 1 & 2
   \end{bmatrix}

Arguments:

   :dims:

      (``string``) Output tensor dimensions

      List of integers

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Uniform:

----------------------------------------
Uniform
----------------------------------------

The :python:`Uniform` layer is a random tensor with a uniform
distribution.

Arguments:

   :min: (``double``) Distribution minimum

   :max: (``double``) Distribution maximum

   :neuron_dims:

      (``string``) Tensor dimensions

      List of integers

   :training_only:

      (``bool``) Only generate random values during training

      If true, the tensor is filled with the distribution mean during
      evaluation.

:ref:`Back to Top<transform-layers>`

________________________________________


.. _Unpooling:

----------------------------------------
Unpooling
----------------------------------------

The :python:`Unpooling` layer is the transpose of the pooling
layer. It is required that the pooling layer be set as the hint layer.

.. warning::
   This has not been well maintained and is probably broken.

.. todo::
   GPU support.

Arguments:

   :num_dims:

      (``int64``) Number of spatial dimensions

      The first data dimension is treated as the channel dimension,
      and all others are treated as spatial dimensions (recall that
      the mini-batch dimension is implicit).

:ref:`Back to Top<transform-layers>`

________________________________________


.. _WeightedSum:

----------------------------------------
WeightedSum
----------------------------------------

The :python:`WeightedSum` layer adds tensors with scaling factors.

Arguments:

   :scaling_factors: (``string``) List of
                     floating-point numbers, one for each input tensor.

:ref:`Back to Top<transform-layers>`

________________________________________


.. _WeightsLayer:

----------------------------------------
WeightsLayer
----------------------------------------

The :python:`WeightsLayer` outputs values from a weights
tensor. Interfaces with a ``weights`` object.

Arguments:

   :dims:

      (``string``) Weights tensor dimensions

      List of integers

:ref:`Back to Top<transform-layers>`

________________________________________


.. _CategoricalRandom:

----------------------------------------
CategoricalRandom (Deprecated)
----------------------------------------

The :python:`CategoricalRandom` layer is deprecated.

Arguments: None

:ref:`Back to Top<transform-layers>`

________________________________________


.. _DiscreteRandom:

----------------------------------------
DiscreteRandom (Deprecated)
----------------------------------------

The :python:`DiscreteRandom` layer is deprecated.

Arguments:

   :values: (``string``)

   :dims: (``string``)

:ref:`Back to Top<transform-layers>`
