.. role:: python(code)
          :language: python

.. role:: c(code)
          :language: c

.. _summarize-images-callback:

============================================================
Summarize Images Callback
============================================================

The purpose of this callback is to output images into an event file at
the end of each epoch, according to the specified intervals. The
images in the event file are displayed using `Tensorboard
<https://www.tensorflow.org/tensorboard>`_. This callback could be
used, for example, to display categorized images or images generated
by an autoencoder or by a GAN.

The method of selecting images, and the layers from which images are
displayed, can be controlled via the :python:`selection_strategy`
argument to the callback. Images that match some boolean value may be
selected with :python:`CategoricalAccuracyStrategy`. A canonical
example of this would be to output images that are classified
incorrectly by a classification network. Alternatively, a fixed number
of images can be displayed using
:python:`TrackSampleIDsStrategy`. This may be used, for example, to
visualize the progress in training a GAN or an autoencoder.

---------------------------------------------
Execution Points
---------------------------------------------

+ After each testing/validation minibatch

---------------------------------------------
Callback Arguments (Python Front-End)
---------------------------------------------

+ :python:`selection_strategy`: The image selection
  strategy. Currently supported options are:

  - :doc:`TrackSampleIDsStrategy <selection_strategy/track_sample_ids_strategy>`
  - :doc:`CategoricalAccuracyStrategy <selection_strategy/categorical_accuracy_strategy>`

+ :python:`image_source_layer_name`: The name of the layer from which
  images will be pulled. A Python Front-End layer's name can be
  accessed via the :python:`name` attribute. This may be the input
  layer, if the true image is requested, or it may be any layer that
  outputs a valid image tensor. This means it must be either
  a 2-D tensor (greyscale image) or a 3-D tensor with the channel
  dimension equal to 1 or 3 (greyscale or RGB, respectively).

+ :python:`epoch_interval`: Epoch frequency to output images. The
  default value is 1; that is, perform the output every epoch.


---------------------------------------------
Examples Using Summarize Images Callback
---------------------------------------------

Python Front-End
--------------------

.. _sample_id_strategy_example:

Track Sample IDs Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: There is currently no built-in way to print the
          original images using a single callback instance. As a
          work-around, if the original image is desired, add a
          second instance of the :python:`CallbackSummarizeImages`
          with the :python:`image_source_layer_name` field set to
          the input layer's name and the :python:`epoch_interval`
          field set to be larger than the total number of epochs you
          expect to run (so it will only output from epoch 0 and
          never again).

.. code-block:: python

    # Set up image selection strategy
    img_strategy = lbann.TrackSampleIDsStrategy(
                     input_layer_name="input",
                     num_tracked_images=10)

    # Pass parameters to callback
    summarize_images = lbann.CallbackSummarizeImages(
                         selection_strategy=img_strategy,
                         image_source_layer_name="reconstruction",
                         epoch_interval=5)

    # Optional- Output original image from input layer once using
    #           a high epoch interval
    summarize_input_layer = lbann.CallbackSummarizeImages(
                              selection_strategy=img_strategy,
                              image_source_layer_name="input",
                              epoch_interval=10000)

.. _cat_acc_strategy_example:

Categorical Accuracy Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set up categorical accuracy layer
    accuracy = lbann.CategoricalAccuracy(prediction_scores, labels)

    # Set up image selection criteria
    match_type = lbann.CategoricalAccuracyStrategy.MatchType

    # Set up image selection strategy
    img_strategy = lbann.CategoricalAccuracyStrategy(
                     cat_accuracy_layer_name=accuracy.name,
                     match_type.NOMATCH,
                     num_images=10)

    # Pass parameters to callback
    summarize_images = lbann.CallbackSummarizeImages(
                         selection_strategy=img_strategy,
                         image_source_layer_name=images.name,
                         epoch_interval=5)


Profotext (Advanced)
----------------------

Track Sample IDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: guess

   callback {
     summarize_images {
       selection_strategy {
         track_sample_ids {
           input_layer_name: "input"
           num_tracked_images: 10
         }
         image_source_layer_name: "reconstruction"
         epoch_interval: 1
       }
     }
   }


Categorical Accuracy Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: guess

   # Set up categorical accuracy layer
   layer {
    parents: "prob"
    parents: "label"
    name: "accuracy"
    data_layout: "data_parallel"
    categorical_accuracy {}
   }

   # Set up callback
   callback {
     summarize_images {
       selection_strategy {
         categorical_accuracy {
           cat_accuracy_layer_name: "accuracy"
           num_images: 10
         }
         image_source_layer_name: "images"
         epoch_interval: 1
         img_format: ".jpg"
       }
     }
   }

.. toctree::
   :hidden:

   selection_strategy/categorical_accuracy_strategy.rst
   selection_strategy/track_sample_ids_strategy.rst
