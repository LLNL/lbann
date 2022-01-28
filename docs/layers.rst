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
appropriate arguments and passing them as a list into the
model. More information about LBANN's layers can be found in
:ref:`layers-list`. See :ref:`layer-options` for layer parameter option.
For example, the input layer, relu layer, and mean squared error layer
could be included with the following:

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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Profobuf (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   layer {
     name: "images"
     children: "relu mse"
     input {
       data_field: "samples"
     }
   }
   layer {
     name: "labels"
     input {
       data_field: "labels"
     }
   }
   layer {
     name: "relu"
     parents: "images"
     children: "mse"
     relu {
     }
   }
   layer {
     name: "mse"
     parents: "images relu"
     mean_squared_error {
     }
   }

------------------------------------------------
Layer Arguments
------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Layer Arguments

   Layer Arguments <layers/layer_args>

------------------------------------------------
LBANN Layers List
------------------------------------------------

.. toctree::
   :maxdepth: 2

   I/O Layers <layers/io_layers>
   Operator Layers <layers/operator_layers>
   Transform Layers <layers/transform_layers>
