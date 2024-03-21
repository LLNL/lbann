import lbann
import torch
import lbann.torch
import lbann.contrib.launcher
import lbann.contrib.args
import google.protobuf.text_format as txtf
import lbann.contrib.hyperparameter as hyper
import argparse
import sys
import os
from os.path import abspath, dirname, join

print()

# ==============================================
# Setup 
# ==============================================

# Debugging
torch._dynamo.config.verbose=True

# Command-line arguments
desc = ('Construct and run a grid search on HRRL PROBIES image and spectra data. ')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='probiesNet', type=str,
    help='scheduler job name (default: probiesNet)')
parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help='mini-batch size (default: 32)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int, 
    help='number of epochs (default: 100)', metavar='NUM')

lbann.contrib.args.add_optimizer_arguments(parser)
args = parser.parse_args()

# Make script with 2 nodes
script = lbann.launcher.make_batch_script(nodes=2, procs_per_node=4) 

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

def make_search_model(
        learning_rate,
    	beta1, 
        beta2,
        eps,
        intermed_fc_layers,
        activation,
        dropout_percent,
        num_labels=201):

    # ==============================================
    # Make TRAIN data reader
    # ==============================================

    cur_dir = dirname(abspath(__file__))

    def construct_train_data_reader():
        import os.path
        module_file = os.path.abspath(__file__)
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        module_dir = os.path.dirname(module_file)

        # Base data reader message
        message = lbann.reader_pb2.Reader()

        # Training set data reader
        data_reader = message
        data_reader.name = 'python'
        data_reader.role = 'train'
        data_reader.shuffle = True
        data_reader.fraction_of_data_to_use = 1.0 
        data_reader.validation_fraction = 0.0 
        data_reader.python.module = 'data.spectra_probies_train_data_info' 
        data_reader.python.module_dir = module_dir
        data_reader.python.sample_function = 'get_sample'
        data_reader.python.num_samples_function = 'num_samples'
        data_reader.python.sample_dims_function = 'sample_dims'

        return message

    # ==============================================
    # Make VAL data reader
    # ==============================================


    def construct_val_data_reader():
        """Construct Protobuf message for Python data reader.

        The Python data reader will import this Python file to access the
        sample access functions.

        """
        import os.path
        module_file = os.path.abspath(__file__)
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        module_dir = os.path.dirname(module_file)

        # Base data reader message
        message = lbann.reader_pb2.Reader()

        # Training set data reader
        data_reader = message
        data_reader.name = 'python'
        data_reader.role = 'validation'
        data_reader.shuffle = True
        data_reader.fraction_of_data_to_use = 1.0 
        data_reader.validation_fraction = 0 
        data_reader.python.module = 'data.spectra_probies_val_data_info'
        data_reader.python.module_dir = module_dir
        data_reader.python.sample_function = 'get_sample'
        data_reader.python.num_samples_function = 'num_samples'
        data_reader.python.sample_dims_function = 'sample_dims'

        return message

    # ==============================================
    # Make TESTING data reader
    # ==============================================


    def construct_test_data_reader():
        """Construct Protobuf message for Python data reader.

        The Python data reader will import this Python file to access the
        sample access functions.

        """
        import os.path
        module_file = os.path.abspath(__file__)
        module_name = os.path.splitext(os.path.basename(module_file))[0]
        module_dir = os.path.dirname(module_file)

        # Base data reader message
        message = lbann.reader_pb2.Reader()

        # Training set data reader
        data_reader = message
        data_reader.name = 'python'
        data_reader.role = 'test'
        data_reader.shuffle = True
        data_reader.fraction_of_data_to_use = 1.0 
        data_reader.validation_fraction = 0 
        data_reader.python.module = 'data.spectra_probies_test_data_info' 
        data_reader.python.module_dir = module_dir
        data_reader.python.sample_function = 'get_sample'
        data_reader.python.num_samples_function = 'num_samples'
        data_reader.python.sample_dims_function = 'sample_dims'

        return message

    import models.probiesNetLBANN_grid_search as model

    images_and_spectra = lbann.Input(data_field='samples') 
    split_results = lbann.Slice(images_and_spectra, axis=0, slice_points=[0,90000,90201]) #should be between the images and spectra

    images = lbann.Identity(split_results)
    responses = lbann.Identity(split_results)

    num_labels = 201

    images = lbann.Reshape(images, dims=[1, 300, 300])

    pred = model.PROBIESNetLBANN(num_labels, intermed_fc_layers, activation, dropout_percent)(images)


    # ==============================================
    # Metrics
    # ==============================================

    # MSE loss between responses and preds 
    mse = lbann.MeanSquaredError([responses, pred])

    layers = list(lbann.traverse_layer_graph([images, responses]))

    # Append metrics
    metrics = [lbann.Metric(mse, name='mse')]

    callbacks = [lbann.CallbackPrint(),
                lbann.CallbackTimer()]
                # for printing the results.  Sample syntax below for dumping
                # multiple layer outputs.  Layer names must be checked for the 
                # particular model in question. 
                # lbann.CallbackDumpOutputs(layers='layer4 layer25 layer46 layer67 pred_out_instance1',execution_modes='test')]

    layers = list(lbann.traverse_layer_graph([images, responses]))

    model = lbann.Model(args.num_epochs,
                        layers=layers, 
                        objective_function=mse, 
                        metrics=metrics,
                        callbacks=callbacks) 
    
    # Setup optimizer 
    opt = lbann.Adam(learn_rate=learning_rate,beta1=beta1,beta2=beta2,eps=eps)

    train_reader = construct_train_data_reader()
    val_reader = construct_val_data_reader()
    test_reader = construct_test_data_reader()
    python_reader = lbann.reader_pb2.DataReader(reader=[train_reader, val_reader, test_reader])

    # Aliases for simplicity
    SGD = lbann.BatchedIterativeOptimizer
    RPE = lbann.RandomPairwiseExchange

    # Construct the local training algorithm
    local_sgd = SGD("local sgd", num_iterations=10)

    # Construct the metalearning strategy. 
    meta_learning = RPE(
        metric_strategies={'mse': RPE.MetricStrategy.LOWER_IS_BETTER})

    # Setup vanilla trainer
    trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)
    return model, opt, python_reader, trainer

# Run the grid search using make_search_model.  Options below.
hyper.grid_search(
    script,
    make_search_model,
    use_data_store=False, # must be False for images to spectra training
    procs_per_trainer=1,
    learning_rate=[0.00001],
    beta1=[0.9],
    beta2=[0.9],
    eps=[1e-8],
    intermed_fc_layers = [[960,240]],
    activation = [lbann.Relu],
    dropout_percent = [0.7])

# Sample syntax for a larger search:
# hyper.grid_search(
#     script,
#     make_search_model,
#     use_data_store=False,
#     procs_per_trainer=1,
#     learning_rate=[0.00001],
#     beta1=[0.9,0.99],
#     beta2=[0.9,0.99],
#     eps=[1e-8],
#     intermed_fc_layers = [[960,240],[1920,960,480,240],[480,240]],
#     activation = [lbann.Relu,lbann.Softmax,lbann.LeakyRelu],
#     dropout_percent = [0.3, 0.5, 0.7])

