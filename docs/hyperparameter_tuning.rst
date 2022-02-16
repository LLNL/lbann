.. role:: python(code)
          :language: python

.. _hyperparameter_tuning:

============================================================
Hyperparameter Tuning
============================================================

.. note:: See the docstring for
          `lbann.contrib.hyperparameter.grid_search` for the API. This
          page provides additional guidance on usage.

.. note:: This feature is experimental and its API is still being
          established. Once it has stabilized, the `hyperparameter`
          submodule should be moved out of the `lbann.contrib`
          submodule and into the top-level `lbann` module.

When training machine learning models, the optimal hyperparameters are
usually unknown and some experimentation is required to achieve
acceptable results. To simplify this process,
`lbann.contrib.hyperparameter.grid_search` implements a naive grid
search algorithm. In particular it launches LBANN with multiple
trainers and trains a separate model on each. If the number of
candidate models is greater than the number of trainers, then LBANN is
run multiple times. Afterwards the log file can be analyzed to
determine the ideal hyperparameter configuration.

The run configuration (e.g. the number of MPI ranks, the work
directory) is specified with a
`lbann.launcher.batch_script.BatchScript` object. See
`lbann.launcher.make_batch_script` for more details on how to
configure a batch job script.

Models are configured with a user-provided function that takes
hyperparameters and returns the objects needed for an LBANN
experiment. Specifically, this function should return a `lbann.Model`,
`lbann.Optimizer`, `lbann.reader_pb2.DataReader`, and `lbann.Trainer`,
i.e. the inputs to `lbann.run`. To specify which values to pass into
this `make_experiment` function, pass lists as keyword arguments into
`lbann.contrib.hyperparameter.grid_search`: the values in the lists
will be forwarded to the corresponding keyward argument in the
`make_experiment` function. The resulting hyperparameter
configurations will be dumped out in a CSV file (typically
`hyperparameters.csv` in the work directory).

Note that this grid search functionality is very flexible. Depending
on how the user defines the `make_experiment` function, it is possible
to train diverse models with different training algorithms and
different datasets. However, it is preferable in practice to train
models with roughly similar performance properties to avoid
load-balancing issues. This is an effective approach to run the LTFB
algorithm: it generates an initial population of diverse models that
can then be subject to an evolutionary process.

Simple Example
------------------------------

Suppose we want to tune the hyperparameters for a simple logistic
classifier. We can define the following function to construct the
model and related objects:

.. code-block:: python

    def make_logistic_classifier(
            learning_rate=0.1,
            bias=False,
            num_labels=10):

        # Construct model
        x = lbann.Input(data_field='samples')
        y = lbann.FullyConnected(x, num_neurons=num_labels, has_bias=bias)
        y_pred = lbann.Softmax(y)
        y_true = lbann.Input(data_field='labels')
        z = lbann.CrossEntropy(y_pred, y_true)
        model = lbann.Model(
            epochs=1,
            layers=lbann.traverse_layer_graph(z),
            objective_function=z)

        # Construct other objects
        optimizer = lbann.SGD(learn_rate=learning_rate)
        data_reader = make_data_reader()
        trainer = lbann.Trainer(mini_batch_size=32)

        return model, optimizer, data_reader, trainer

Then we can perform a grid search with:

.. code-block:: python

    script = lbann.launcher.make_batch_script(nodes=2, procs_per_node=4)
    lbann.contrib.hyperparameter.grid_search(
        script,
        make_logistic_classifier,
        procs_per_trainer=1,
        learning_rate=[0.5, 0.1, 0.05, 0.01],
        bias=[True, False])

This will generate and train eight models. Since we are running with
eight trainers (8 MPI ranks and 1 rank per trainer), we only need to
run LBANN once. If we ran with four trainers instead, LBANN would run
twice. It is not necessary to perfectly line up the number of models
and the number of trainers, but there could be idle compute nodes if
they don't match.
