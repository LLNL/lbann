.. _sec:hdf5_data_reader:

================
HDF5 Data Reader
================

The HDF5 data reader is a flexible data reader that is designed to
ingest data samples from HDF5 format and place them into Conduit nodes
within LBANN. Rather than having to rewrite the data reader for any
given sample format, the HDF5 data reader uses a pair of *Schema*
files to allow the users to specify both the structure of a sample for
a given data set, as well as which fields of the sample to use for any
given experiment.

=======================
HDF5 Schema files
=======================

This section describes the two *Schemas* that users must supply to run
LBANN with HDF5 data. A *Schema* is a yaml description of the hdf5
data hierarchy, along with additional transformational information
(e.g, normalization values).

The *data_schema.yaml* file contains a description of the hdf5 data
hierarchy as it exists on disk.  The *experiment_schema.yaml* file
contains a specification of a subset of the data that is to be used in
an experiment.  We refer to these as the the data fields.
(*data_schema.yaml* and *experiment_schema.yaml* are standins for the
user's filenames of choice.)

.. note:: it follows from above that the
  *data_schema* need only
  describe that portion of the data set that is of interest to a
  common set of experiments. But, there is no penalty for supplying a
  schema with excess information.

An example *data_schema* follows.

.. code-block:: yaml

 inputs:
    initial_modes:
    trans_u:
    trans_v:
        metadata:
            scale: 1.666669
            bias: 0.5000008
            ordering: 104
 outputs:
    scalars:
        BWx:
        BT:
        tMAXt:
        MT:
            B4:
            after:
    images:
        metadata:
            dims: [64, 64]
            channels: 4
            scale: [29.258502, 858.26596, 100048.72, 4807207.0]
        img_1:
        img_2:
        img_3:


The "metadata" nodes are not part of the hdf5 hierarchy, but are
optionally added by the user.  For brevity, we only show two metadata
nodes. Metadata nodes:

1. are optional

2. contain transformational information supplied by the user (i.e, are not part of the HDF5 hierarchy)

3. are inherited from their parents

Regarding the last point, internally, every regular node (node from
the hdf5 hierarchy) contains a metadata node as a child. Where not
explicitly stated in the schema, empty nodes are
created. Algorithmically, a node inherits its parent's
metadata. However, if both contain the same field name, the child's
value prevails. We'll call this the "trickle-down rule". Hence, in the
above, the metadata *dims*, *channels*, and *scale* will be applied to
all three images (e.g, "image" is "img_1's" parent, and the "metadata"
node belongs to the parent (is the parent's child).

Prior to discussing metadata further, here is an example *experiment_schema*.

.. code-block:: yaml

 inputs:
     metadata:
        pack: datum
 outputs:
     metadata:
        pack: datum
     scalars:
         MT:
     images:

As stated above, this schema specifies the data fields to be used in
an experiment. Algorithmically, the data fields are determined thusly:
a tree traversal is conducted on the *experiment_schema*. If a leaf in
the experiment_schema is an internal node in the *data_schema*, then
the traversal is "continued" at that point in the *data_schema* (of
course, bringing metadata nodes along).

The following data fields are thus available per the *experiment_schema*:

1. inputs: initial_modes, trans_u, trans_v
2. outputs/scalars: MT/B4, MT/after
3. outputs/images: imag_1, img_2, img3

A point to note is that, because the user specified
outputs/scalars/MT, we only "continue the transveral" for the MT child
of the outputs/scalars node; i.e, we do not traverse the BWx, BT, or
tMAXt child nodes.

A primary design consideration of the two-schema plan was to enable
users to easily alter the selection, ordering, and transformations of
input data. In this regard, the *data_schema* will most likely be
static, i.e, it's metadata contains directives that are unlikely to
change from experiment to experiment; think: normalization values. The
*experiment_schema* can be thought of as a more minimalist approach to
specifying data fields and metadata. That said, users have
considerable latitude as to how and where they specify metadata; just
bear in mind the trickle-down rule.

-------------------
Metadata Directives
-------------------

By *Metadata Directive*, or more simply *directive*, we refer to the
keys in the metadata nodes, which we group as follows.

1. packing - the *pack* directive requests the concatanation of
   multiple data fields. The resulting(composite) field can be
   retrieved by a call that contains the directive's value, which must
   be one of datum, label, response.  The *ordering* directives(below)
   determine the order in which concatanation occurs. All data fields
   in a packing group must be of the same primitive datatype. If not,
   ensure that they are *coerced* (below)
   # REVIEWERS: SHOULD WE RELAX
   # THIS? Ie, specifying the type in one place, and let the coercion
   # happen automagically?


2. ordering - the *ordering* directive is a numeric field that
   determines how data is packed. This directive lets the user
   determine "the order in which things are stuffed into the tensor."
   The directive's values need be neither consecutive nor
   unique. Advice: this optional field is perhaps best placed in the
   *data_schema*, with desired over-rides in the
   *experiment_schema*. Use widely spaced numbers in the *data_schema*
   so you can easily over-ride (rearrange your data) in the
   experiment_schema.

3. normalization - we recognize the two numeric directives: *scale*
   and *bias*, which have their usual meanings. The values should be
   scalars or, for images, etc, lists of scalars.

4. coercing - the *coerce* directive transforms data from its original
   type (i.e, as stored on media) to some other type, which is stored
   in memory and available upon request.  By example, if there's a
   "foo" data field on disk, of type float64_array, and the metadata
   contains "coerce: float32", then the data will be converted to a
   float32_array. Note that a *coerce* directive's value refers to a
   primitive scalar type; all data fields are assumed to be scalars or
   arrays of scalars (arrays, aka: 1D tensors, vectors, lists,
   etc). One effect of our example is a reduction in memory use,
   though coercing in the other direction would have increased
   memory. As mentioned above, coercion may be necessary in
   conjunction with *pack* directives.

5. images - in addition to the *scale* and *bias* directives, images
   may contain *dims*, *channels*, and *hwc* directives. If the *hwc*
   directive specifies the images will be converted from
   height-width-channel encoding to some other format; at present, the
   only transformational format we support is channel-height-width.

--------------
Larger Example
--------------

We conclude this section with a more fleshed-out example of the schemas.

*data_schema*:

.. code-block:: yaml

 inputs:
   shape_model_initial_modes:(4,3):
     metadata:
       scale: 1.666672
       bias: 0.5
       ordering: 100
   betti_prl15_trans_u:
     metadata:
       scale: 1.000002
       bias: -1.603483e-07
       ordering: 101
   betti_prl15_trans_v:
     metadata:
       scale: 1.000001
       bias: -1.406672e-06
       ordering: 102
 outputs:
   scalars:
     BWx:
       metadata:
         scale: 7.610738
         bias: -0.4075375
         ordering: 201
     BT:
       metadata:
         scale: 1.459875
         bias: -3.427656
         ordering: 202
     tMAXt:
       metadata:
         scale: 1.490713
         bias: -3.495498
         ordering: 203
     BWn:
       metadata:
         scale: 43.75123
         bias: -1.593477
         ordering: 204
   images:
     metadata:
       dims: [64, 64]
       channels: 4
       scale: [29.258502, 858.26596, 100048.72, 4807207.0]
       bias: [0.0, 0.0, 0.0, 0.0]
       hwc: "chw"

     (0.0, 0.0):
       0.0:
         emi:
           metadata:
             ordering: 300
     (90.0, 0.0):
       0.0:
         emi:
           metadata:
             ordering: 301

*experiment_schema*:

.. code-block:: yaml

 inputs:
   metadata:
     pack: "datum"

 outputs:
   metadata:
     pack: "datum"

   scalars:
     BWx:
       metadata:
         ordering: 555
     BT:
       metadata:
         ordering: 554

   images:
     metadata:
       coerce: "double"
     (90.0, 0.0):
       0.0:
