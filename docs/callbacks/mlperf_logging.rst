.. role:: python(code)
          :language: python

.. role:: c(code)
          :language: c

.. _mlperf-logging-callback:

============================================================
MLPerf Logging Callback
============================================================

The MLPerf callback exports an MLPerf compatible log for running
benchmarks on LBANN. The logging output is included in the out.log
file located in the LBANN run directory.

---------------------------------------------
Execution Points
---------------------------------------------

+ setup
+ on setup end
+ on epoch begin
+ on epoch end
+ on train begin
+ on train end
+ on batch evaluate begin
+ on batch evaluate end

.. _callback-arguments:

---------------------------------------------
Callback Arguments
---------------------------------------------

   .. note:: While technically optional, omitting arguments will
             result in "UNKNOWN_<FIELD_NAME>" appearing in the log
             results (with the exception of sub_org).

   :sub_benchmark: (``string``) Benchmark name. A list of benchmarks
                   can be found in the `MLPerf Benchmarks Suite
                   <https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#3-benchmarks>`_.

   :sub_org: (``string``, optional) Organization running the
             benchmarks. Default: `LBANN`.

   :sub_division: (``string``) Closed or open division. See `Divisions <https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc#4-divisions`_

   :sub_status: (``string``) Submission status. (onprem, cloud, or
                preview)

   :sub_platform: (``string``) Submission platform/hardware. (Example:
                  Longhorn, NVIDIA DGX A100, JUWELS_Booster)

.. _examples-using-export-onnx:

------------------------------------------------------
Example Using Export ONNX Callback (Python Front-End)
------------------------------------------------------

.. code-block:: python

   # Pass parameters to callback
   mlperf_logging = lbann.CallbackMlperfLogging(
                                   sub_benchmark="SUBMISSION_BENCHMARK",
                                   sub_org="SUBMISSION_ORGANIZATION",
                                   sub_division="SUBMISSION_DIVISION",
                                   sub_status="SUBMISSION_STATUS",
                                   sub_platform="SUBMISSION_PLATFORM")
