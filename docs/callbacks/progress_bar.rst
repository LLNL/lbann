.. role:: python(code)
          :language: python

.. _progress_bar-callback:

============================================================
Print Progress Bar Callback
============================================================

The purpose of this callback is to show progress while training.
The callback will print a bar to the standard output with the current minibatch,
the number of total minibatches, iterations per second (moving average over 10
last minibatches), and an estimation of how long it will take for the epoch to
complete.

Note that this callback will pollute output log files (e.g., ``out.log``),
and if any other callback prints out information during training epochs, the
output may interfere with or erase the current progress bar.

---------------------------------------------
Execution Points
---------------------------------------------

+ Before each training minibatch

---------------------------------------------
Callback Arguments (Python Front-End)
---------------------------------------------

+ :python:`interval`: Frequency in which to print the progress bar. The
  default value is 1; that is, print every minibatch.
