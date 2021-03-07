================
HDF5 Data Reader
================
This section describes the two *Schemas* that users must supply to run LBANN with HDF5 data. A *Schema* is a yaml description of the hdf5 data hierarchy, along with additional transformational information (e.g, normalization values).


The *data_schema.yaml* file contains a description of the hdf5 data hierarchy as it exists on disk. 
The *experiment_schema.yaml* file contains a description of that portion of the data (called the data fields) that is to be used in an experiment (i.e, subset of the experiment_schema.yaml)


.. note:: users are, or course, free to choose their own yaml file names

.. note:: it follows that the
  *data_schema* need only  
  describe that portion of the data set that is of interest to a common set of experiments.

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




For brevity, we only show two  metadata nodes. Metadata nodes:

1. are optional

2. contain information supplied by the user (are not part of the HDF5 hierarchy)

3. are inherited from their parents

Regarding the last point, internally, every regular node   contains a metadata node. If not explicitly stated in the schema, internally, empty metadata nodes are created. Algorithmically, a node inherits its parent's metadata. However, if both contain the same field name, the child's value will prevail. (I like to think: "metada trickles down the tree"). Hence, in the above, the metadata *dims* and *channels* will be applied to all three images.

Prior to discussing metadata further, we present an example of an experiment_schema.

.. code-block:: yaml

 inputs:
     metadata:
        pack: sample
 outputs:
     metadata:
        pack: sample
     scalars:
         MT:
     images:

As stated above, this schema describes the data fields that are to be used in an experiment. Algorithmically, the data fields are determined thusly: a tree traversal is conducted on the *experiment_schema*. If a leaf in the experiment_schema is an internal node int the *data_schema*, then the traversal is "continued" in the *data_schema* (of course, bringing metadata nodes along).

The following data fields are thus available per the *experiment_schema*:

1. inputs: initial_modes, trans_u, trans_v 
2. outputs/scalars: MT/B4, MT/after
3. outputs/images: imag_1, img_2, img3

A point to note is that, because the user specified outputs/scalars/MT, we only "continue the transveral" for the MT child of the outputs/scalars node; i.e, we do not traverse the BWx, BT, or tMAXt nodes.

A primary design consideration of the two-schema plan was to enable users to easily alter the selection of input data. In this regard, the *data_schema* should be considered static, i.e, it's metadata contains directives that are unlikely to change from experiment to experiment; think: normalization values.

(That said, you are free to attach any metadata directives to any node, bearing in mind the trickle-down rule.)

-------------------
Metadata Directives
-------------------

By *Metadata Directive*, or more simply *directive*, we refer to the keys in the metadata nodes, which we group as follows.

1. packing - the *pack* directive requests the concatanation of multiple data fields. The (composite) field can be retrieved TODO

2. ordering - the *ordering* directive is a numeric field that determines how data is packed. This directive lets the user determine "the order in which things are stuffed into the tensor." The directive's values need be neither consecutive nor unique. Advice: this optional field is perhaps best placed in the *data_schema*, with desired over-rides in the *experiment_schema*. Use widely spaced numbers in the *data_schema* so you can easily over-ride (rearrange your data) in the experiment_schema.

3. normalization - we recognize the two numeric directives: *scale* and *bias*, which have their usual meanings. The values should be scalars or, for images, etc, lists of scalars.

4. coercing - the *coerce* directive transforms data from its original type (i.e, as stored on media) to some other type. By example, if there's a "phoo" data field on disk, of type float64_array, and the metadata contains "coerce: float32", then when a request is later made to the data_reader for a piece of data called, "phoo," the returned type will be a float32_array. In this case an effect of coercion is a reduction in memory use. As explained above, coercion may be necessary in conjunction with *pack* directives. 

5. images - in addition to *scale* and *bias*,  TODO

