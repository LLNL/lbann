**************************************************
LBANN Software Architecture and Class Overview
**************************************************

Trainers (i.e. execution environment)
******************************************

A trainer is a collection of compute resources and defines a explicit
communication domain.  It provides the execution for both the training
and inference of a trained model.  Once constructed a trainer owns an
LBANN comm object that defines both intra- and inter-trainer
communication domains.  Additionally, a trainer will contain an I/O
thread pool that is used to fetch and pre-process data that will be
provided to the trainer's models.

A trainer owns:

* comm object
* I/O thread pool
* One or more models
* Execution context for each model
* In the future, it will also contain the data readers.

Execution Context
******************************************

When a model is attached to a trainer the execution context of the
training algorithm is stored in an execution_context class (or
sub-class) per execution mode.  Thus there is one execution context
per model and mode that contains all of the state with respect to the
training algorithm being applied to the model.

For example it tracks the current:

* step
* execution mode
* epoch
* and a pointer back to the trainer

Termination Criteria (Pending)
******************************************

(Pending feature) When a model is going to be trained or evaluated,
the termination criteria is specified in an object that is passed into
the training algorithm.  (Note that this feature is under development,
currently the termination criteria is dictated by when the training
algorithm executes a fixed number of epochs.)

Training Algorithms
******************************************

The training algorithm defines the optimization that is to be
applied to the model(s) being trained.  Additionally, it can
specify how to evaluate the model.

Model
******************************************

A model is a collection of operations with dependencies encoded as a
directed acyclic graph (DAG).  In a typical formulation, these
operations form a neural network that will be either trained or used
for inference.  Each operation in the model is an instance of the
layer class.  The model is then a collection of layers that perform
transformations and mathematical operations on data that is passed
between layers.  The model's DAG is executed in topological order.
Inside of some layer types are weight matrices that define a trained
model.  (Note that LBANN should be able to support non-DNN models, but
this is a subject for future work.)

Each layer in the graph contains a set of tensors that holds the
inputs, computed outputs, gradients with respect to the outputs, and
gradients with respect to the inputs.  Furthermore, for each layer in
the graph with learnable parameters, there is an associated weight
tensor that form the learned weights of the model.  The model also
owns the objective function, since that is integrally tied into the
model's computational graph.  Additionally, the model owns both the
default optimizer that is used to provide a standard optimizer for the
model's weight tensors.  Once each weight tensor is instantiated, it
will owns an instance of an optimizer.

The model also owns the max_mini_batch_size that is supported by the
model.  This is due to the fact that it changes the size and shape of
input, output, and gradient tensors.  Additionally, the model owns a
field that controls if background I/O is allowed for this model and
associated data reader.
