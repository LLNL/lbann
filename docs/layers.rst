.. role:: python(code)
          :language: python

.. _layers:

============================================================
Layers
============================================================

Layers in a neural network are arranged as a directed acyclic
graph. They take one input tensor from each parent layer and send
an output tensor to each child layer. Some layers may recieve
tensors from weights objects (trainable parameters).

LBANN performs implicit mini-batching. If the user specifies a
layer to handle 3D image data (in channel-height-width format), it
is stored internally as a 4D tensor (in NCHW format). This scheme
implies that computation is mostly independent between mini-batch
samples (with a few exceptions like batchnorm).

The default value for fields in protobuf messages is zero-like
(false for bool, empty string for string). Thus, all defaults are
zero-like unless otherwise stated.

.. _using-layers:

------------------------------------------------
Using Layers
------------------------------------------------

Layers are used by adding them to the python front end with the
appropriate arguments and passing them as a list into the model. More
information about LBANN's layers can be found in
:ref:`layers-list`. See :ref:`layer-arguments` for a description of
layer parameters.  For example, the input layer, relu layer, and mean
squared error layer could be included with the following:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Front End
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   images = lbann.Input(data_field='samples', name="images")
   labels = lbann.Input(data_field='labels', name="labels")

   relu = lbann.Relu(images, name="relu")
   mse = lbann.MeanSquaredError([images, relu], name="mse")

   layer_list = list(lbann.traverse_layer_graph([images, labels]))

   model = lbann.Model(num_epochs,
                       layer_list,
                       objective_function,
                       metrics,
                       callbacks)

.. _layer-arguments:

-------------------------------------------
Common Layer Arguments
-------------------------------------------

  :name:

     (``string``, optional) default = ``layer<index>``

     Unique identifier for layer

     Must not contain spaces

  :parents:

     (``Iterable of Layer``, optional) Parent layers, i.e. Sources
     of input tensors

     List of layer names

  :children:

     (``Iterable of Layer``, optional) Child layers, i.e. Destinations
     for output tensors

     List of layer names

  :weights:

     (``Iterable of Weights``, optional) Weights objects.

     Typically used as trainable parameters

     List of weights names

  :device_allocation:

     (``string``, optional) Data tensor device

     Options: CPU or GPU

     If LBANN has been built with GPU support, default is
     GPU. Otherwise, CPU

  :datatype:

     (``lbann.DataType``, optional)

     Data type used for activations and weights

-------------------------------------------
  Advanced Layer Options
-------------------------------------------

  :hint_layer:

     (``Layer``, optional) Hint layer for configuring output
     dimensions

     Typically used to specify that a layer has the same output
     dimensions as another.

  :data_layout:

     (``string``, optional) Data tensor layout

     Options: data_parallel (default) or model_parallel

  :parallel_strategy:

     (``dictionary``, optional) Configuration for advanced
     parallelization strategies

     Parallel Strategy Options:

        :sample_groups: (``int64``)

        :sample_splits: (``int64``)

        :height_groups: (``int64``)

        :height_splits: (``int64``)

        :width_groups: (``int64``)

        :width_splits: (``int64``)

        :channel_groups: (``int64``)

        :channel_splits: (``int64``)

        :filter_groups: (``int64``)

        :filter_splits: (``int64``)

        For fully-connected layers:

        :replications: (``int64``)

        :procs_per_replica: (``int64``)

        :depth_groups: (``int64``)

        :depth_splits: (``int64``)

     Sub-grid parallelism:

        :sub_branch_tag: (``int64``)

        :sub_branch_resource_percentage: (``int64``)

        :enable_subgraph: (``bool``)

-------------------------------------------
  Deprecated Layer Options
-------------------------------------------

Deprecated:

  :freeze: (``bool``)


.. _layers-list:

------------------------------------------------
LBANN Layers List
------------------------------------------------

.. toctree::
   :maxdepth: 2

   I/O Layers <layers/io_layers>
   Operator Layer <layers/operator_layer>
   Transform Layers <layers/transform_layers>
   Learning Layers <layers/learning_layers>
   Loss Layers <layers/loss_layers>
   Math Layers <layers/math_layers>
   Regularization Layers <layers/regularization_layers>
   Activation Layers <layers/activation_layers>
   Image Layers <layers/image_layers>
   Miscellaneous Layers <layers/miscellaneous_layers>
