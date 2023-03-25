.. role:: python(code)
          :language: python

.. role:: c(code)
          :language: c

.. _export-onnx-callback:

============================================================
Export ONNX Callback
============================================================

Running with this callback exports a trained model in `ONNX
<https://onnx.ai/>`_ format. The default ONNX filename is
:python:`lbann.onnx`. An optional debug string can also be
printed to by setting debug string filename argument. These options
can be controlled using the :ref:`callback-arguments`. The files are
created in the LBANN run directory.

---------------------------------------------
Execution Points
---------------------------------------------

+ On train end

.. _callback-arguments:

---------------------------------------------
Callback Arguments
---------------------------------------------

   :output_filename: (``string``, optional) Default value:
                 ``lbann_output.onnx``.

   :debug_string_filename: (``string``, optional) Name of debug string
                           file. If not set, the debug string is not
                           output.

.. _examples-using-export-onnx:

------------------------------------------------------
Example Using Export ONNX Callback (Python Front-End)
------------------------------------------------------

.. code-block:: python

   # Pass parameters to callback
   export_onnx = lbann.CallbackExportOnnx(
                   output_filename="model.onnx",
                   debug_string_filename="debug_onnx.txt")
