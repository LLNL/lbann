.. role:: python(code)
          :language: python

==============================
Categorical Accuracy Strategy
==============================

----------
Summary
----------

The :python:`CategoricalAccuracyStrategy` is used to view a snapshot
of images in the dataset being used in the training session that match
a boolean criterion. To simplify things in the model construction,
this strategy can print images whose output is :python:`true`, images
whose output is :python:`false`, or all images.  A canonical use-case
is to print the images that are (in)correctly categorized by a
classification model. The number of images output is limited by a
user-provided parameter or until no more matches are found.

.. note:: The name of this class erroneously suggests a rather narrow
          use-case. We are looking to change the name in a future
          release of LBANN. In fact, this strategy can take as input
          any boolean layer, not just categorical accuracy layers.

----------
Arguments
----------

+ :python:`categorical_accuracy_layer_name` (string): The name of the
  boolean layer to be used to determine matches. A Python Front-End
  layer's name can be accessed via the :python:`name` attribute. A
  common use-case is the name of a :python:`CategoricalAccuracy` layer
  that has been added to a model.

+ :python:`match_type`
  (:python:`lbann.CategoricalAccuracyStrategy.MatchType`): Criterion for
  selecting images to output. Possible values are:

  =================  =======================================================
  :python:`NOMATCH`  Output images corresponding to :python:`false` values.
  :python:`MATCH`    Output images corresponding to :python:`true` values.
  :python:`ALL`      Output all images.
  =================  =======================================================

  The default value is :python:`NOMATCH`.
  
+ :python:`num_images_per_epoch` (uint): The maximum number of images to
  output per epoch. The default value is 10.

----------
Usage
----------

See the :ref:`usage example<cat_acc_strategy_example>` as part of
the :doc:`CallbackSummarizeImages </callbacks/summarize_images>`
documentation.


