.. role:: python(code)
          :language: python

.. role:: c(code)
          :language: c

.. _export-onnx-callback:

============================================================
Export ONNX Callback
============================================================

Running with this callback exports a trained model in `ONNX
<https://onnx.ai/>`_ format. The default file name is
:python:`lbann_output.onnx`. An optional debug string can also be
printed to :python:`onnx_debug.txt`. These options can be controlled
using the :ref:`callback-arguments`. The files are created in the
LBANN run directory.

---------------------------------------------
Execution Points
---------------------------------------------

+ On train begin

.. _callback-arguments:
---------------------------------------------
Callback Arguments (Python Front-End)
---------------------------------------------

   :print_debug_string:

      (``bool``, optional) Default value: False.

      Print debug string to a text file.

   :output_file: (``string``, optional) Default value:
                 ``lbann_output.onnx``.

.. _examples-using-export-onnx:

---------------------------------------------
Examples Using Export ONNX Callback
---------------------------------------------

Python Front-End
--------------------

.. code-block:: python

   # Pass parameters to callback
   export_onnx = lbann.CallbackExportOnnx(
                   print_debug_string=True,
                   output_file="model.onnx")

Prototext (Advanced)
----------------------

.. code-block:: guess

   callback {
     summarize_images {
       print_debug_string: true
       output_file: "model.onnx"
     }
   }
