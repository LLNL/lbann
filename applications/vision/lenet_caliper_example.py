import argparse
import lbann
import data.mnist
import lbann.contrib.args
import lbann.contrib.launcher
import importlib
# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Train LeNet on MNIST data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_lenet', type=str,
    help='scheduler job name (default: lbann_lenet)')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Input data
images = lbann.Input(data_field='samples')
labels = lbann.Input(data_field='labels')

# LeNet
x = lbann.Convolution(images,
                      num_dims = 2,
                      num_output_channels = 6,
                      num_groups = 1,
                      conv_dims_i = 5,
                      conv_strides_i = 1,
                      conv_dilations_i = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = "max")
x = lbann.Convolution(x,
                      num_dims = 2,
                      num_output_channels = 16,
                      num_groups = 1,
                      conv_dims_i = 5,
                      conv_strides_i = 1,
                      conv_dilations_i = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = "max")
x = lbann.FullyConnected(x, num_neurons = 120, has_bias = True)
x = lbann.Relu(x)
x = lbann.FullyConnected(x, num_neurons = 84, has_bias = True)
x = lbann.Relu(x)
x = lbann.FullyConnected(x, num_neurons = 10, has_bias = True)
probs = lbann.Softmax(x)

# Loss function and accuracy
loss = lbann.CrossEntropy(probs, labels)
acc = lbann.CategoricalAccuracy(probs, labels)

# ----------------------------------
# Setup experiment
# ----------------------------------
#
# Use calipermode dict with one of these string arguments to return callback function with apropo arguments
#  'normal' : sets skip_init = False,autotune=False
#  'normaL_skip_init' : set skip_init = True, autotune=False
#  'capture_autotune' : sets skip_init=False, autotune=True
#  'apply_autotune : sets skip_init=False, autotune=False, tuned_omp_thread = int(autotune_threads)
#
# capture autotune will run a timing loop around im2col_2d and save key im2col_threads in the output lbann.cali file
#
# for apply_autotune determine if we have capability to readback capture autotune results via 
# caliperreader module in caliper-reader python files
# caliper-reader can be installed using pip install caliper-reader
# or alternatively, add python/caliper-reader path in the cloned Caliper repo to PYTHONPATH
#
# This autotune example relies on a local save of lbann.cali - a prior output collected with
# lbann.CallbackProfilerCaliper(skip_init = False, autotune = True)
#
# Then in this file we readback the autotuned openmp im2col_threads and apply it as
# tuned_omp_threads argument in
# lbann.CallbackProfilerCaliper(skip_init = False, autotune = False, tuned_omp_threads = int(autotune_threads))
#
# To take a quick look at the Caliper output run cali-query -T lbann.cali 
# 
# To see if lbann.cali has Adiak key im2col_threads for the autotune apply example run:
# cali-query -G lbann.cali | grep im2col


def caliper_normal():
    return lbann.CallbackProfilerCaliper(skip_init=False,autotune=False)

def caliper_normal_skip():
    return lbann.CallbackProfilerCaliper(skip_init=True,autotune=False)

def caliper_capture_autotune():
    return lbann.CallbackProfilerCaliper(skip_init=False,autotune=True)

def caliper_apply_autotune():
    reader_spec = importlib.util.find_spec("caliperreader")
    reader_found = reader_spec is not None

    if reader_found:
      print("Caliper Reader found")
    try: 
      cr = reader_spec.loader.load_module()
      gg = cr.read_caliper_globals('lbann.cali')
      autotune_threads = gg['im2col_threads']
    except:
      print("Can't apply autotune")
      autotune_threads = 0

    return lbann.CallbackProfilerCaliper(skip_init=False,autotune=False,tuned_omp_threads = int(autotune_threads))

calipermode = {
        'normal' : caliper_normal,
        'normal_skip' : caliper_normal_skip,
        'capture_autotune': caliper_capture_autotune,
        'apply_autotune' : caliper_apply_autotune,
        }



# Setup model
autotune_threads = 0
mini_batch_size = 64
num_epochs = 3 
model = lbann.Model(num_epochs,
                    layers=lbann.traverse_layer_graph([images, labels]),
                    objective_function=loss,
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrintModelDescription(),
                               lbann.CallbackPrint(),
                               calipermode['normal'](),
                               lbann.CallbackTimer()])

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Setup data reader
data_reader = data.mnist.make_data_reader()

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

# ----------------------------------
# Run experiment
# ----------------------------------
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
