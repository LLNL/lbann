import argparse

import lbann
import lbann.contrib.args
import lbann.contrib.launcher

import model
import data

# ----------------------------------
# Command-line arguments
# ----------------------------------

# Parse command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_motif')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of passes over dataset (default: 100)', metavar='NUM')
parser.add_argument(
    '--work-dir', action='store', default=None, type=str,
    help='working directory', metavar='DIR')
parser.add_argument(
    '--batch-job', action='store_true',
    help='submit as batch job')
args = parser.parse_args()

# Hard-coded options
data_dim = 1234
latent_dim = 64
learn_rate = 0.025

# ----------------------------------
# Construct LBANN objects
# ----------------------------------

trainer = lbann.Trainer(
    mini_batch_size=args.mini_batch_size,
    num_parallel_readers=0,
)
model_ = model.make_model(
    data_dim,
    latent_dim,
    args.num_epochs,
)
optimizer = lbann.SGD(learn_rate=learn_rate, momentum=0.9)
data_reader = data.make_data_reader()

# ----------------------------------
# Run
# ----------------------------------

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(
    trainer,
    model_,
    data_reader,
    optimizer,
    job_name=args.job_name,
    work_dir=args.work_dir,
    batch_job=args.batch_job,
    **kwargs,
)
