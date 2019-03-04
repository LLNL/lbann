Trainers (i.e. execution environment):

  A trainer is a collection of compute resources and defines a explicit
  communication domain.  It provides the execution for both the training
  and inference of a trained model.  Once constructed a trainer owns an
  LBANN comm object that defines both intra- and inter-trainer
  communication domains.  Additionally, a trainer will contain an I/O
  thread pool that is used to fetch and pre-process data that will be
  provided to the trainer's models.

  A trainer owns:

  comm object
  I/O thread pool
  One or more models
  Execution context for each model

  <In the future it will also contain the data readers>

Execution Context:

  When a model is attached to a trainer the execution context is stored
  in an execution_context class (or sub-class) per execution mode.  Thus
  there is one execution context per model and mode that contains all of
  the state with respect to the training or execution context.

  For example it  tracks the current:
  step
  execution mode
  epoch
  and a pointer back to the trainer

Termination Criteria:

  When a model is going to be trained or evaluated, the termination
  criteria is specified in an object that is passed into the training
  algorithm.

Training Algorithms:

  The training algorithm defines the optimization that is to be
  applied to the model(s) being trained.  Additionally, it can
  specify how to evaluate the model.

Model:

  A model is the neural network that will be either trained or used
  for inference.  It is a collection of layers that perform
  transformations and mathematical operations on data that is passed
  between layers.  Layers are composed in a general directed acyclic
  graph (DAG) and executed in a depth-first traversal.  Inside of some
  layer types are weight matrices that define a trained model.

  The model also owns the objective function, since that is integrally
  tied into the model's computational graph.  Additionally, it owns both
  the default optimizer and each weight matrix owns an instance of the
  optimizer.

  The model also owns the max_mini_batch_size that is supported by the
  model.  This is due to the fact that it changes the size and shape of
  weight matrices.  Additionally, the model owns if background I/O is
  allowed.
