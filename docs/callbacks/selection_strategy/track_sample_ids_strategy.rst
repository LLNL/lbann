.. role:: python(code)
          :language: python

==============================
Track Sample IDs Strategy
==============================

----------
Summary
----------

The :python:`TrackSampleIDsStrategy` selection strategy is used by
:python:`CallbackSummarizeImages` to output a constant set of images
over the duration of a training run of LBANN.  Use of this strategy is
ideally suited to generative applications, as it allows users to
visualize the ability of a network to reproduce the same image over
time.

----------
Arguments
----------

+ :python:`input_layer_name`: the name of the input layer with the
  original images. For reasons inherent to the C++ code, this must be
  an :python:`Input` layer. A Python Front-End layer's name can be
  accessed via the :python:`name` attribute.

+ :python:`num_tracked_images`: the number of images to track. If
  unset, 10 images will be tracked. This is a proxy for the user
  specifying images to track based on some unique identifier. We are
  considering methods to expose this functionality; this is work in
  progress.

----------
Usage
----------

See the :ref:`usage example<sample_id_strategy_example>` as part of
the :doc:`CallbackSummarizeImages </callbacks/summarize_images>`
documentation.

