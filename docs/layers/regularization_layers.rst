.. role:: python(code)
          :language: python


.. _regularization-layers:

====================================
Regularization Layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`BatchNormalization`, "Channel-wise batch normalization"
   :ref:`Dropout`, "Probabilistically drop tensor entries"
   :ref:`EntrywiseBatchNormalization`, "Entry-wise batch normalization"
   :ref:`InstanceNorm`, "Normalize over data channels"
   :ref:`LayerNorm`, "Normalize over data samples"
   :ref:`LocalResponseNormalization`, "Local-response normalization"
   :ref:`SeluDropout`, "Scaled dropout for use with SELU activations"

________________________________________


.. _BatchNormalization:

----------------------------------------
BatchNormalization
----------------------------------------

The :python:`BatchNormalization` layer performs channel-wise batch
normalization.

Each input channel is normalized across the mini-batch to have zero
mean and unit standard deviation. Learned scaling factors and biases
are then applied. This uses the standard approach of maintaining the
running mean and standard deviation (with exponential decay) for use
at test time.

This layer maintains four weights: scales, biases, running means, and
running variances. Each has a size equal to the number of channels. In
order to disable the affine operation, manually construct weights
without optimizers.

See:

Sergey Ioffe and Christian Szegedy. "Batch Normalization: Accelerating
Deep Network Training by Reducing Internal Covariate Shift." In
International Conference on Machine Learning, pp. 448-456. 2015.

Arguments:

   :decay:

      (``double``, optional) Decay factor for running statistics

      Default: 0.9

   :epsilon:

      (``double``, optional) Small number for numerical stability

      Default: 1e-5

   :statistics_group_size:

      (``int64``, optional) Size of process group for computing
      statistics

      Default: 1

      A group size of 1 implies purely local statistics. A negative
      group size indicates global statistics (i.e. statistics over the
      entire mini-batch).

Deprecated arguments:

   :stats_aggregation: (``string``)

Deprecated and unused arguments:

   :scale_init: (``double``)

   :bias_init: (``double``)

:ref:`Back to Top<regularization-layers>`

________________________________________


.. _Dropout:

----------------------------------------
Dropout
----------------------------------------

The :python:`Dropout` layer probabilistically drops tensor entries.

The values are multiplied by 1/(keep probability) at training time. Keep
probabilities of 0.5 for fully-connected layers and 0.8 for input
layers are good starting points. See:

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever,
and Ruslan Salakhutdinov. "Dropout: a simple way to prevent neural
networks from overfitting." The Journal of Machine Learning Research
15, no. 1 (2014): 1929-1958.

Arguments:

   :keep_prob:

      (``double``) Probability of keeping each tensor entry

      Recommendation: 0.5

:ref:`Back to Top<regularization-layers>`

________________________________________


.. _EntrywiseBatchNormalization:

----------------------------------------
EntrywiseBatchNormalization
----------------------------------------

The :python:`EntrywiseBatchNormalization` layer performs entry-wise
batch normalization.

Each input entry is normalized across the mini-batch to have zero mean
and unit standard deviation. This uses the standard approach of
maintaining the running mean and standard deviation (with exponential
decay) for use at test time.

This layer maintains two weights: running means, and running
variances. Each has a shape identical to the data tensor. It is common
to apply an affine operation after this layer, e.g. with the
entry-wise scale/bias layer.

See:

Sergey Ioffe and Christian Szegedy. "Batch Normalization: Accelerating
Deep Network Training by Reducing Internal Covariate Shift." In
International Conference on Machine Learning, pp. 448-456. 2015.

Arguments:

   :decay:

      (``double``) Decay factor for running statistics

      Recommendation: 0.9

   :epsilon:

      (``double``) Small number for numerical stability

      Recommendation: 1e-5

:ref:`Back to Top<regularization-layers>`

________________________________________


.. _InstanceNorm:

----------------------------------------
InstanceNorm
----------------------------------------

The :python:`InstanceNorm` layer normalizes data samples over data
channels.

Each channel within a data sample is normalized to have zero mean and
unit standard deviation. See:

Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. "Instance
normalization: The missing ingredient for fast stylization." arXiv
preprint arXiv:1607.08022 (2016).

This is equivalent to applying layer normalization independently to
each channel. It is common to apply an affine operation after this
layer, e.g. with the channel-wise scale/bias layer.

Arguments:

    :epsilon:

        (``google.protobuf.DoubleValue``, optional) Small number to avoid
        division by zero.

        Default: 1e-5

:ref:`Back to Top<regularization-layers>`

________________________________________


.. _LayerNorm:

----------------------------------------
LayerNorm
----------------------------------------

The :python:`LayerNorm` layer normalizes data samples.

Each data sample is normalized to have zero mean and unit standard
deviation. See:

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer
normalization." arXiv preprint arXiv:1607.06450 (2016).

It is common to apply an affine operation after this layer, which is possible
to enable with the ``scale`` and ``bias`` arguments.

By default, the entire data sample is normalized. The argument ``start_dim``
controls the normalized shape and dimensions by specifying a subset of
tensor dimensions to normalize. For example, a value of -1 will normalize only
the last tensor dimension.

Arguments:

    :epsilon:

        (``google.protobuf.DoubleValue``, optional) Small number to
        avoid division by zero.

        Default: 1e-5

    :scale:

        (``bool``, optional) Apply an elementwise learned scaling factor
        after normalization.

        Default: false

    :bias:

        (``bool``, optional) Apply an elementwise learned bias after
        normalization.

        Default: false

    :start_dim:

        (``int64``, optional) The tensor dimension to start normalizing from.
        All dimensions including the given one will be normalized. A value of 0
        normalizes each data sample in its entirety. Negative numbers
        are also permitted.

        Default: 0


:ref:`Back to Top<regularization-layers>`

________________________________________


.. _LocalResponseNormalization:

----------------------------------------
LocalResponseNormalization
----------------------------------------

The :python:`LocalResponseNormalization` layer normalizes values
within a local neighborhood.

See:

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet
classification with deep convolutional neural networks." In Advances
in Neural Information Processing Systems, pp. 1097-1105. 2012.

Arguments:

   :window_width: (``int64``)

   :lrn_alpha: (``double``)

   :lrn_beta: (``double``)

   :lrn_k: (``double``)

:ref:`Back to Top<regularization-layers>`

________________________________________


.. _SeluDropout:

----------------------------------------
SeluDropout
----------------------------------------

The :python:`SeluDropout` layer is a scaled dropout for use with SELU
activations.

A default keep probability of 0.95 is recommended. See:

Gunter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp
Hochreiter. "Self-normalizing neural networks." In Advances in Neural
Information Processing Systems, pp. 971-980. 2017.

Arguments:

   :keep_prob: (``double``) Recommendation: 0.95

   :alpha: (``double``, optional) Default:
           1.6732632423543772848170429916717


   :scale: (``double``, optional) Default:
           1.0507009873554804934193349852946


:ref:`Back to Top<regularization-layers>`
