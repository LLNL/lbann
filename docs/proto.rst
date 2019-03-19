Protobuf
=================================

LBANN uses the Tensorflow protobuf format for specifying the
architecture of neural networks, data readers, and optimizers.  It
serves as the "assembly language" interface to the toolkit.  The
python front end of LBANN will emit a network description in the
protobuf format that is ingested at runtime.

.. autodoxygenindex::
 :project: proto
