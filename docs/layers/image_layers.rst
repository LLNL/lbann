.. role:: python(code)
          :language: python


.. _image-layers:

====================================
Image layers
====================================

.. csv-table::
   :header: "Layer", "Description"
   :widths: auto

   :ref:`BilinearResize`, "Resize image with bilinear interpolation"
   :ref:`CompositeImageTransformation`, "Rotate a image clockwise
   around its center, then shear , then translate"
   :ref:`Rotation`, "Rotate a image clockwise around its center"

________________________________________


.. _BilinearResize:

----------------------------------------
BilinearResize
----------------------------------------

Resize image with bilinear interpolation

Expects a 3D input tensor, which is interpreted as an image in CHW
format. Gradients are not propagated during backprop.

Arguments:

   :height: (``int64``) Output image height

   :width: (``int64``) Output image width

:ref:`Back to Top<image-layers>`

________________________________________


.. _CompositeImageTransformation:

----------------------------------------
CompositeImageTransformation
----------------------------------------

Rotate a image clockwise around its center, then shear , then
translate

Expects 4 inputs: a 3D image tensor in CHW format, a scalar rotation
angle, a tensor for (X,Y) shear factor, a tensor  for (X,Y) translate.

Arguments: None

:ref:`Back to Top<image-layers>`

________________________________________


.. _Rotation:

----------------------------------------
Rotation
----------------------------------------

Rotate a image clockwise around its center

Expects two inputs: a 3D image tensor in CHW format and a scalar
rotation angle.

Arguments: None

:ref:`Back to Top<image-layers>`

________________________________________
