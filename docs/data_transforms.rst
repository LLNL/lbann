Data Transformation Pipeline
==============================

LBANN supports a large collection of data transformations as part of
its preprocessing pipeline.

General Transformations
------------------------------

The following transformations are always enabled in LBANN and can be
used for any type of data.


Normalize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Sample Normalize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Entries in the sample are scaled to a mean of 0 and standard deviation
of 1. This transformation takes no parameters.

.. math::

    x_i = (x_i - \mu) / \sigma,

where :math:`\mu` is the mean and :math:`\sigma` is the standard
deviation of the values in the sample.

We don't actually expose these well in the Python Front-end, but if
you felt so compelled, you could modify the base data reader prototext
you're using by adding a message like this:

.. code-block:: none

   transforms {
        sample_normalize {
        }
    }


Scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This transformation scales the data entrywise by a fixed constant:

.. math::

    x_i = \alpha x_i

Example prototext is:

.. code-block:: none

    transforms {
        scale {
            scale: 0.1
        }
    }

This will scale each data sample entrywise to ten percent of the
original value.


Scale and Translate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Image Transformations
------------------------------

Some transformations are only applicable to image data. These
transformations require that LBANN be built with OpenCV support
(:code:`+vision` in the Spack specification).


Adjust Brightness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Adjust Contrast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Adjust Saturation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Center Crop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Change Image Tensor Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. repack_HWC_to_CHW_layout.hpp
TODO


Color Jitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Colorize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Cutout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Grayscale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Horizontal Flip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Normalize To LBANN Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Random Affine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Random Crop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Random Resized Crop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Random Resized Crop With Fixed Aspect Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Resize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenCV is required to use this transformation. Each sample is resized
up or down to the given size. An example usage is:

.. code-block:: none

    transforms {
        resize {
            height: 123
            width: 321
        }
    }

Bilinear interpolation is used for interpolation.


Resized Center Crop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


To LBANN Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Vertical Flip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO
