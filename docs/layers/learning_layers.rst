.. role:: python(code)
          :language: python


.. _learning-layers:

====================================
Learning Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`ChannelwiseFullyConnected`, "Apply affine transformation to
   tensor channels"
   :ref:`ChannelwiseScaleBias`, "Apply per-channel scale and bias"
   :ref:`Convolution`, "Convolution"
   :ref:`Deconvolution`, "Deconvolution"
   :ref:`Embedding`, "Lookup table to embedding vectors"
   :ref:`EntrywiseScaleBias`, "Apply entry-wise scale and bias"
   :ref:`FullyConnected`, "Affine transformation"
   :ref:`GRU`, "Stacked gated recurrent unit"

________________________________________


.. _ChannelwiseFullyConnected:

----------------------------------------
ChannelwiseFullyConnected
----------------------------------------

The :python:`ChannelwiseFullyConnected` layer applies an affine
transformation to tensor channels.

The input tensor is sliced along the first tensor dimension (the
"channel" dimension for image data in CHW format) and the same affine
transformation is applied to each slice. Following a row-vector
convention:

.. math::

   y(i,*) = \text{vec}( x(i,*) ) W^T + b

Two weights are required if bias is applied: the linearity and the
bias. Only the linearity weights are required if bias is not
applied. If weights aren't provided, the linearity weights are
initialized with He normal initialization and the bias weights are
initialized to zero.

Arguments:

   :output_channel_dims: (``repeated uint64``) Output tensor
                         dimensions, excluding the channel dimension

   :bias:

       (``google.protobuf.BoolValue``, optional) Whether to apply bias

       Default: ``True``

   :transpose:

       (``google.protobuf.BoolValue``, optional) Whether to apply
       transpose of weights matrix

       Default: ``False``

:ref:`Back to Top<learning-layers>`

________________________________________


.. _ChannelwiseScaleBias:

----------------------------------------
ChannelwiseScaleBias
----------------------------------------

The :python:`ChannelwiseScaleBias` layer applies per-channel scale and
bias. The input tensor is sliced along the first tensor dimension (the
"channel" dimension, assuming image data in CHW format) and scale and
bias terms are applied independently to each slice. More precisely,
given input and output tensors
:math:`X,Y\in\mathbb{R}^{d_1\times\cdots\times d_n}` and scale and
bias vectors :math:`a,b\in\mathbb{R}^{d_1}`:

.. math::

   Y_{i,j,\cdots} = a_i X_{i,j,\cdots} + b_i

The scale and bias vectors are fused into a single weights tensor to
reduce the number of gradient allreduces during backprop. In
particular, the weights tensor is a
:math:`\text{num_channels} \times 2` matrix, where the first column
corresponds to scale terms and the second column to bias terms.

Arguments: None

:ref:`Back to Top<learning-layers>`

________________________________________


.. _Convolution:

----------------------------------------
Convolution
----------------------------------------

The :python:`Convolution` layer applies convolution (more precisely,
cross-correlation) to the input tensor. This is primarily optimized
for image data in CHW format.

Two weights are required if bias is applied: a kernel tensor (in KCHW
format) and per-channel biases. Only the kernel weights are required
if bias is not applied. If weights aren't provided, the kernel weights
are initialized with He normal initialization and the bias weights are
initialized to zero.

Arguments:

   :num_dims:

       (``int64``) Number of spatial dimensions

       The first data dimension is treated as the channel dimension, and
       all others are treated as spatial dimensions (recall that the
       mini-batch dimension is implicit).

   :out_channels:

       (``int64``) Channel dimension of output tensor

   :kernel_size:

       (``list[int64]`` or ``int64``) Convolution kernel dimensions

       List of integers, one for each spatial
       dimension.

   :padding:

       (``list[int64]`` or ``int64``) Convolution padding

       List of integers, one for each spatial
       dimension.

   :stride:

       (``list[int64]`` or ``int64``) Convolution strides

       List of integers, one for each spatial
       dimension. Used when ``has_vectors`` is enabled.

   :dilation:

       (``list[int64]`` or ``int64``) Convolution dilations

       List of integers, one for each spatial dimension. Defaults to
       dilations of 1, i.e. undilated convolution.


   :has_bias: (``bool``) Whether to apply per-channel bias

   :groups:

      (``int64``, optional) Number of channel groups for grouped
      convolution

      Default: 1

   :conv_tensor_op_mode:

      (``ConvTensorOpsMode``) Special behavior with FP16 tensor cores

      Ignored for non-GPU layers.

:ref:`Back to Top<learning-layers>`

________________________________________


.. _Deconvolution:

----------------------------------------
Deconvolution
----------------------------------------

The :python:`Deconvolution` layer is the transpose of standard deep
learning convolution.

Pedantic comments: this operation is commonly called "deconvolution"
in the deep learning community, but it is not a true deconvolution.
Also, the "convolution" operation commonly used in the deep learning
is actually cross-correlation.

Arguments:

   :num_dims: (``int``): Number of spatial dimensions

   :out_channels: (``int``): Channel dimension of output tensor

   :kernel_size: (``list[int]`` or ``int``): Convolution kernel dimensions

   :stride: (``list[int]`` or ``int``): Convolution stride

   :padding: (``list[int]`` or ``int``): Convolution padding

   :output_padding: (``list[int]`` or ``int``): Padding for output tensor.
     The output tensor size is ambiguous when the convolution is
     strided. If this is not set, then we will output the smallest
     valid output tensor.

   :groups: (``int``): Number of convolution groups (default: 1)

   :has_bias: (``bool``): Whether to apply channel-wise bias (default: True)

   :dilation: (``list[int]`` or ``int``): Convolution dilation (default: 1)

   :conv_tensor_op_mode:

      (``ConvTensorOpsMode``) Special behavior with FP16 tensor cores

      Ignored for non-GPU layers.

:ref:`Back to Top<learning-layers>`

________________________________________


.. _Embedding:

----------------------------------------
Embedding
----------------------------------------

The :python:`Embedding` layer is a lookup table to embedding vectors.

Takes a scalar input, interprets it as an index, and outputs the
corresponding vector. The number of embedding vectors and the size of
vectors are fixed. If the index is out-of-range, then the output is a
vector of zeros.

The embedding vectors are stored in an
:math:`\text{embedding_dim} \times \text{num_embeddings}` weights
matrix. Note that this is the transpose of the weights in the PyTorch
embedding layer.

   :num_embeddings: (``int64``) Size of dictionary of embeddings

   :embedding_dim: (``int64``) Size of embedding vectors

   :padding_idx: (``google.protobuf.Int64Value``) If the index is set,
                 then the corresponding vector is initialized with
                 zeros. The function gradient w.r.t. this embedding
                 vector always

:ref:`Back to Top<learning-layers>`

________________________________________


.. _EntrywiseScaleBias:

----------------------------------------
EntrywiseScaleBias
----------------------------------------

The :python:`EntrywiseScaleBias` layer applies entry-wise scale and
bias.

Scale and bias terms are applied independently to each tensor
entry. More precisely, given input, output, scale, and bias tensors
:math:`X,Y,A,B\in\mathbb{R}^{d_1\times\cdots\times d_n}`:

.. math::

   Y = A \circ X + B

The scale and bias terms are fused into a single weights tensor to
reduce the number of gradient allreduces during backprop. In
particular, the weights tensor is a :math:`\text{size} \times 2`
matrix, where the first column correspond to scale terms and the
second column to bias terms.

Arguments: None

:ref:`Back to Top<learning-layers>`

________________________________________


.. _FullyConnected:

----------------------------------------
FullyConnected
----------------------------------------

The :python:`FullyConnected` layer is an affine transformation.

Flattens the input tensor, multiplies with a weights matrix, and
optionally applies an entry-wise bias. Following a row-vector
convention:

.. math::

   y = \text{vec}(x) W^T + b

Two weights are required if bias is applied: the linearity and the
bias. Only the linearity weights are required if bias is not
applied. If weights aren't provided, the linearity weights are
initialized with He normal initialization and the bias weights are
initialized to zero.

For flat data, this layer is similar to Keras' dense layer or
PyTorch's linear operation. However, it implicitly flattens
multi-dimensional data. To avoid this flattening, consider the
channel-wise fully-connected layer.

Arguments:

   :num_neurons: (``int64``) Output tensor size

   :has_bias: (``bool``) Whether to apply entry-wise bias

   :transpose: (``bool``) Whether to apply transpose of weights

:ref:`Back to Top<learning-layers>`

________________________________________


.. _GRU:

----------------------------------------
GRU
----------------------------------------

The :python:`GRU` layer is a stacked gated recurrent unit.

Expects two inputs: a 2D input sequence (
:math:`\text{sequence_length}\times\text{input_size}`) and a 2D
initial hidden state (
:math:`\text{num_layers}\times\text{hidden_size}`).

Uses four weights per GRU cell: "ih\_matrix" (
:math:`3 \text{hidden_size}\times\text{input_size}` for layer 0 and
:math:`3 \text{hidden_size}\times\text{hidden_size}` for other layers),
"hh\_matrix" (:math:`3 \text{hidden_size}\times\text{hidden_size}`),
"ih_bias" (:math:`3 \text{hidden_size}`), "hh_bias"
(:math:`3 \text{hidden_size}`).

Support is experimental and requires either cuDNN (on GPU) or oneDNN
(on CPU).

    .. todo:: Support bidirectional RNNs

Arguments:

   :hidden_size: (``uint64``) Size of each hidden state and output vector

   :num_layers:

      (``google.protobuf.UInt64Value``, optional) Number of stacked GRU
      cells

      Default: 1

:ref:`Back to Top<learning-layers>`
