.. role:: python(code)
          :language: python


.. _miscellaneous-layers:

====================================
Miscellaneous Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`Argmax`, "Get index of maximum-value tensor entry"
   :ref:`Argmin`, "Get index of minimum-value tensor entry"
   :ref:`ChannelwiseMean`, "Mean values across channel dimension"
   :ref:`ChannelwiseSoftmax`, "Softmax across channel dimension"
   :ref:`Covariance`, "Covariance between entries of two tensors"
   :ref:`DistEmbedding`, "Embedding layer with distributed weights"
   :ref:`External`, "Create layer from an external library"
   :ref:`MiniBatchIndex`, "Position of data sample within mini-batch"
   :ref:`MiniBatchSize`, "Size of current mini-batch"
   :ref:`OneHot`, "Convert index to a one-hot vector"
   :ref:`RowwiseWeightsNorms`, "L2 norm of each row of a weights matrix"
   :ref:`UniformHash`, "Apply a hash function to get uniformly
   distributed values"
   :ref:`Variance`, "Variance of tensor entries"

________________________________________


.. _Argmax:

----------------------------------------
Argmax
----------------------------------------

The :python:`Argmax` layer gets the index of the maximum-value tensor
entry.

Expects a 1D input tensor. If multiple entries have the same maximum
value, outputs the index of the first one.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _Argmin:

----------------------------------------
Argmin
----------------------------------------

The :python:`Argmin` layer gets the index of the minimum-value tensor
entry.

Expects a 1D input tensor. If multiple entries have the same minimum
value, outputs the index of the first one.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _ChannelwiseMean:

----------------------------------------
ChannelwiseMean
----------------------------------------

The :python:`ChannelwiseMean` layer computes mean values across
channel dimensions.

The input tensor is sliced along the first tensor dimension (the
"channel" dimension for image data in CHW format) and the mean value
is computed for each slice.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _ChannelwiseSoftmax:

----------------------------------------
ChannelwiseSoftmax
----------------------------------------

The :python:`ChannelwiseSoftmax` layer applies the Softmax function
across channel dimensions.

The input tensor is sliced along the first tensor dimension (by default, the
"channel" dimension for image data in CHW format) and the softmax
function is computed for each slice.

Arguments:

   :dim: (``int64``) The tensor dimension to use (defaults to first).

   :single_dim_mode: (``bool``) If true, only performs softmax on the chosen
                     dimension. Otherwise all dimensions but ``dim`` will be
                     used.

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _Covariance:

----------------------------------------
Covariance
----------------------------------------

The :python:`Covariance` layer computes the covarience between entries
of two tensors.

Arguments:

   :biased: (``bool``) Use biased estimator, i.e. sample covariance

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _DistEmbedding:

----------------------------------------
DistEmbedding
----------------------------------------

The :python:`DistEmbedding` layer is the embedding layer with
distributed weights.

This is similar to the embedding layer, which takes integer indices
and returns embedding vectors from a lookup table. However, the
embedding vectors are distributed between processes and one-sided
inter-process communication is performed with OpenSHMEM (on CPU) or
NVSHMEM (on GPU).

The main benefit of this model-parallel approach is to handle cases
where the embedding vectors don't fit on one process. It should also
have better scaling properties when the mini-batch size is very large.

To take advantage of sparse gradients, the distributed embedding layer
provides the option to bypass the optimizer (which currently only
supports dense gradients) and perform sparse SGD directly on the
embedding weights. If enabled, SGD occurs during the layers "update"
phase (i.e. in the virtual update_compute function). Otherwise, the
layer converts sparse gradients to a dense tensor and passes it into
the usual optimizer. This is a hack and will be deprecated once the
optimizer class supports sparse gradients.

.. warning:: This is experimental.

.. todo:: Sparse SGD with optimizer class

Arguments:

   :num_embeddings: (``int64``) Size of dictionary of embeddings.

   :embedding_dim: (``int64``) Size of embedding vectors.

   :sparse_sgd:

      (``bool``) Perform sparse SGD during backprop.

      Bypasses optimizer class.

   :learning_rate: (``double``) SGD learning rate.

   :barrier_in_forward_prop:

      (``bool``) Perform a blocking barrier a the beginning of forward
      prop.

      This layer performs synchronization with non-blocking barriers
      to ensure the correctness of asynchronous communication. However,
      gradient checking changes the embedding values without performing
      any synchronization. The quickest fix is to do a blocking barrier
      at the beginning of forward prop to make sure that all the
      embeddings are ready to be accessed.

      .. todo:: Think of a way to avoid this synchronization.

:ref:`Back to Top<miscellaneous-layers>`


.. _External:

----------------------------------------
External
----------------------------------------

The :python:`External` layer creates a layer from an external
library.

An external layer can be created by compiling an LBANN layer object in
a separate shared library (such as an .so file), along with a setup
function that creates it. This layer accepts a file path and a
layer name (so more than one can exist in a library), and
will invoke the library dynamically to create the layer. The layer
in the external library can be set with an arbitrary number of inputs,
outputs, and weights.

Compiling a layer only needs to include the LBANN headers and link against
``liblbann.so``. An ``extern "C"`` function named ``setup_<LAYER NAME>``
must exist for LBANN to be able to create the layer.

.. warning::
   Make sure you link the library with the version of LBANN you plan to
   run it with.


.. note:: An example layer can be found in ``src/layers/unit_test/example_layer.cpp``.

Arguments:

   :filename: (``string``) Library file name or path.

   :layer_name: (``string``) Layer name for setup function.


:ref:`Back to Top<miscellaneous-layers>`


________________________________________


.. _MiniBatchIndex:

----------------------------------------
MiniBatchIndex
----------------------------------------

The :python:`MiniBatchIndex` is the position of a data sample within a
mini-batch.

LBANN does implicit mini-batching and data samples are usually
processed independently. This layer is helpful if some mini-batch
samples need to be processed differently from others.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _MiniBatchSize:

----------------------------------------
MiniBatchSize
----------------------------------------

The :python:`MiniBatchSize` is the size of the current mini-batch.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _OneHot:

----------------------------------------
OneHot
----------------------------------------

The :python:`OneHot` layer converts an index to a one-hot vector.

Expects a scalar input tensor and outputs a 1D tensor. The input is
interpreted as an index, and output entries are one if they correspond
to that index and zero otherwise. Out-of-range indices are ignored.

Arguments:

   :size: (``int64``) Size of one-hot vector

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _RowwiseWeightsNorms:

----------------------------------------
RowwiseWeightsNorms
----------------------------------------

The :python:`RowwiseWeightsNorms` layer is the L2 norm of each row of
a weights matrix.

.. warning::

   This layer is experimental and finnicky. It is intended for use
   with the matrix weights from a fully-connected layer, and other
   use-cases may have strange behavior.

Given a weights object, this layer computes the L2 norm for each row
of the underlying matrix. Note that the internal matrix may have
different dimensions than the logical weight dimensions.

This layer expects to have one weights object. During setup, that
weights object should be initialized by another layer before this
layer's setup phase. Setting a "hint layer" may be necessary to
enforce this ordering.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _UniformHash:

----------------------------------------
UniformHash
----------------------------------------

The :python:`UniformHash` layer applies a hash function to get
uniformly distributed values.

Each input entry is hashed with MD5 and scaled to [0,1).

.. warning:: Currently only supported on GPU.

Arguments: None

:ref:`Back to Top<miscellaneous-layers>`

________________________________________


.. _Variance:

----------------------------------------
Variance
----------------------------------------

The :python:`Variance` layer computes the variance of tensor entries.

Arguments:

   :biased: (``bool``) Use biased estimator, i.e. sample variance

:ref:`Back to Top<miscellaneous-layers>`
