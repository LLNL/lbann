# This file contains the headers and flavor text to appear on the
# organizational top pages of the developer documentation. In both
# dictionaries, the keys are paths relative to "include/lbann". The
# header will be the title for the page. The flavor text will be
# inserted under the title and before a toctree listing contents.

lbann_rst_headers = {
    '.' : 'LBANN API',
    'callbacks' : 'Callback Interface',
    'data_readers' : 'Data Readers Interface',
    'data_store' : 'Data Store Interface',
    'layers' : 'Layer Interface',
    'layers/activations' : 'Activation Layers',
    'layers/image' : 'Image Layers',
    'layers/io' : 'I/O Layers',
    'layers/learning' : 'Learning Layers',
    'layers/math' : 'Math Layers',
    'layers/misc' : 'Miscellaneous Layers',
    'layers/regularizers' : 'Regularization Layers',
    'layers/transform' : 'Transform Layers',
    'io': 'I/O Utilities',
    'io/data_buffers': 'Data Buffers for Data Ingestion',
    'metrics' : 'Metrics Interface',
    'models' : 'Models Interface',
    'objective_functions' : 'Objective Function Interface',
    'objective_functions/weight_regularization' : 'Objective Functions for Weight Regularization',
    'optimizers' : 'Optimizer Interface',
    'proto' : 'Protobuf and Front-End Utilities',
    'utils' : 'General Utilities',
    'utils/threads' : 'Multithreading Utilities',
    'weights' : 'Weights Interface'
}

lbann_rst_flavor_text = {
    '.' : '''
Welcome to the LBANN developers' documentation. The documentation is
laid out following a similar structure to the source code to aid in
navigation.
    ''',

    'callbacks' : '''
Callbacks give users information about their model as it is trained.
Users can select which callbacks to use during training in their
model prototext file.''',

    'data_readers' : '''
Data readers provide a mechanism for ingesting data into LBANN.  This
is typically where a user may have to interact with the LBANN source
code.''',

    'data_store' : '''
The data store provides in-memory caching of the data set and
inter-epoch data shuffling.''',

    'layers' : '''
LBANN models are defined in model prototext files. The bulk of these
defintions will be the series of layers which make up the model
itself. LBANN layers all inherit from the common base
:code:`lbann::Layer`. The concrete layers belong to one of several
categories.''',

    'io': '''
Classes for persisting the state of LBANN (checkpoint and restart),
 file I/O and data buffers.''',

    'io/data_buffers': '''
The data buffer classes describe how data is distributed across the
input layer. Note that this part of the class hierarchy is scheduled
to be deprecated and folded into the existing input layer class.''',

    'metrics' : '''
A metric function can be used to evaluate the performance of a model
without affecting the training process. Users define the metric with
which to test their model in their model prototext file.
The available metric functions in LBANN are found below.''',

    'models' : '''
A model is a collection of layers that are composed into a
computational graph. The model also holds the weight matrices for each
learning layer. During training the weight matrices are the free
parameters. For a trained network during inference the weight matrics
are preloaded from saved matrices. The model also contains the
objective function and optimizer classes for the weights.''',

    'objective_functions' : '''
An objective function is the measure that training attempts to optimize.
Objective functions are defined in a user's model defintion prototext
file. Available objective functions can be found below.''',

    'objective_functions/weight_regularization' : '''
TODO:Something about objective_functions/weight_regularization''',

    'optimizers' : '''
Optimizer algorithms attempt to optimize model weights. Optimizers
are selected when invoking LBANN via a command line argument
(:code:`--optimizer=<path_top_opt_proto>`). Available optimizers
are found below.''',

    'proto' : '''
LBANN uses the Tensorflow protobuf format for specifying the
architecture of neural networks, data readers, and optimizers. It
serves as the "assembly language" interface to the toolkit. The
python front end of LBANN will emit a network description in the
protobuf format that is ingested at runtime.''',

    'utils' : 'Utility classes and functions.',

    'utils/threads' : 'TODO: Something about utils/threads',

    'weights' : '''
The weight class is the representation of the trainable parameters in
the neural network.  Learning layers each have an independent weight
class.  During stochastic gradient descent training the weight
matrices are updated after each forward and backward propagation step.'''
}
