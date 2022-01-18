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

Models with weights are not currently supported.

---------------------------------------------
Execution Points
---------------------------------------------

+ On train begin

.. _callback-arguments:
---------------------------------------------
Callback Arguments (Python Front-End)
---------------------------------------------

+ :python:`bool print_debug_string`: Default value is false. Set to
  true to print debug string to a text file.

+ :python:`string output_file`: Default value is
  :python:`lbann_output.onnx`.


---------------------------------------------
Examples Using Export ONNX Callback
---------------------------------------------

Python Front-End
--------------------

.. _export_onnx_example:

Export ONNX Python Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Pass parameters to callback
    export_onnx = lbann.CallbackExportOnnx(
                    print_debug_string=True,
                    output_file="model.onnx")

Prototext (Advanced)
----------------------

Export ONNX Prototext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: guess

   callback {
     summarize_images {
       print_debug_string: true
       output_file: "model.onnx"
     }
   }
