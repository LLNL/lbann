.. role:: python(code)
          :language: python

.. _layer-arguments:

===========================================
Common Layer Arguments
===========================================

  :name:

     * (``string``, optional)
     * default = ``layer<index>``
     * Unique identifier for layer
     * Must not contain spaces

  :parents:

     * (``Iterable of Layer``, optional)
     * Parent layers, i.e. Sources of input tensors
     * Space-separated list of layer names

  :children:

     * (``Iterable of Layer``, optional)
     * Child layers, i.e. Destinations of input tensors
     * Space-separated list of layer names

  :weights:

     * (``Iterable of Weights``, optional)
     * Weights objects. Typically used as trainable parameters
     * Space-separated list of weights names

  :device_allocation:

     * (``string``, optional)
     * Data tensor device
     * If LBANN has been built with GPU support, default is
       GPU. Otherwise, CPU

  :datatype:

     * (``lbann.DataType``, optional)
     * Options: CPU or GPU
     * Data type used for activations and weights

===========================================
  Advanced options
===========================================

  :hint_layer:

     * (``Layer``, optional)
     * Hint layer for configuring output dimensions
     * Typically used to specify that a layer has the same output
       dimensions as another.

  :data_layout:

     * (``string``, optional)
     * Data tensor layout
     * Options: data_parallel (default) or model_parallel

  :parallel_strategy:

     * (``dictionary``, optional)
     * Configuration for advanced parallelization strategies

===========================================
  Deprecated options
===========================================

* Deprecated

  * ``bool num_neurons_from_data_reader``
  * ``bool freeze``

* Deprecated and unused

  * ``repeated WeightsData weights_data``
  * ``string top``
  * ``string bottom``
  * ``string type``
