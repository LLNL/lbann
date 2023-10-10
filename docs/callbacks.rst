.. role:: python(code)
          :language: python

.. _callbacks:

============================================================
Callbacks
============================================================

LBANN has numerous callbacks that can be used to collect
data about an experiment, such as scalars, metrics, weights,
memory usage, images, etc. The documentation of many of these
is pending; see the :ref:`list of Available
Callbacks<available-callbacks>` for the documented ones.

The callbacks are set to execute at various times, and can be
used to display images according to either a boolean output or
their global sample index.

For a complete listing of callbacks and details about their
functionality, please see :ref:`Available
Callbacks<available-callbacks>`.

.. _using-callbacks:

------------------------------------------------
Using Callbacks
------------------------------------------------

Callbacks are used by adding them to the python front end with the
appropriate arguments and passing them as a list into the model.
For example, the callbacks timer, print_statistics, and save model
could be included with the following:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python Front End
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   timer = lbann.CallbackTimer()
   print_stats = lbann.CallbackPrintStatistics(
                 batch_interval=5)
   save_model = lbann.CallbackSaveModel(
                dir=".",
                disable_save_after_training=True)

   callbacks = [timer,
                print_stats,
                save_model]

   model = lbann.Model(num_epochs,
                       layers,
                       objective_function,
                       metrics,
                       callbacks)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Profobuf (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   callback {
     timer {
     }
     print_statistics {
       batch_interval: 5
     }
     save_model {
       dir: "."
       disable_save_after_training: true
     }
   }

.. _available-callbacks:

------------------------------------------------
Available Callbacks
------------------------------------------------

.. toctree::
   :maxdepth: 1

   Export Onnx <callbacks/export_onnx>
   Summarize images <callbacks/summarize_images>
   Print progress bar <callback/progress_bar>

..
   Alternate updates <callbacks/alternate_updates>
   Check dataset <callbacks/check_dataset>
   Check gradients <callbacks/check_gradients>
   Check init <callbacks/check_init>
   Check metric <callbacks/check_metric>
   Check nan in activation values <callbacks/check_nan>
   Check matrices in small values <callbacks/check_small>
   Checkpoint <callbacks/checkpoint>
   Confusion matrix <callbacks/confusion_matrix>
   Debug <callbacks/debug>
   Debug io <callbacks/debug_io>
   Dump error signals <callbacks/dump_error_signals>
   Dump gradients <callbacks/dump_gradients>
   Dump minibatch sample indices <callbacks/dump_minibatch_sample_indices>
   Dump outputs <callbacks/dump_outputs>
   Dump weights <callbacks/dump_weights>
   Early stopping <callbacks/early_stopping>
   Gpu memory usage <callbacks/gpu_memory_usage>
   Hang <callbacks/hang>
   Learning rate <callbacks/learning_rate>
   Load model <callbacks/load_model>
   Ltfb <callbacks/ltfb>
   Mixup <callbacks/mixup>
   Monitor io <callbacks/monitor_io>
   Perturb adam <callbacks/perturb_adam>
   Perturb dropout <callbacks/perturb_dropout>
   Perturb learning rate  <callbacks/perturb_learning_rate>
   Print model description <callbacks/print_model_description>
   Print statistics <callbacks/print_statistics>
   Profiler <callbacks/profiler>
   Replace weights <callbacks/replace_weights>
   Save images <callbacks/save_images>
   Save model <callbacks/save_model>
   Save topk models <callbacks/save_topk_models>
   Summary <callbacks/summary>
   Sync layers <callbacks/sync_layers>
   Timeline <callbacks/timeline>
   Timer <callbacks/timer>
   Variable minibatch <callbacks/variable_minibatch>
   Compute model size <callbacks/compute_model_size>
