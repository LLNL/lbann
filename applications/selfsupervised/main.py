#!/usr/bin/env python3
import os.path
import argparse
import random
import lbann
import lbann.contrib.launcher
import lbann.contrib.args
import lbann.proto
import classifier
import pretrain_siamese
import util

# Paths
current_dir = os.path.dirname(os.path.realpath(__file__))

# ==============================================
# Options
# ==============================================

# Command-line options
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_siamese')
parser.add_argument(
    '--pretrain', action='store', default='siamese', type=str,
    help='pretraining model (default: siamese)')
parser.add_argument(
    '--num-patches', action='store', default=3, type=int,
    help='number of patches and Siamese heads (default: 3)', metavar='NUM')
parser.add_argument(
    '--pretrain-epochs', action='store', default=20, type=int,
    help='number of pretraining epochs (default: 20)', metavar='NUM')
parser.add_argument(
    '--batch-job', action='store_true',
    help='submit script as batch job')
parser.add_argument(
    '--checkpoint-interval', action='store', default=0, type=int,
    help='epoch frequency for checkpointing')
args = parser.parse_args()

# ==============================================
# Setup experiment
# ==============================================

# Pretraining model
if not args.pretrain or args.pretrain == 'siamese':
    model1, reader1, opt1 = pretrain_siamese.setup(
        num_patches=args.num_patches,
        mini_batch_size=512,
        num_epochs=args.pretrain_epochs,
        learning_rate=0.005,
        checkpoint_interval=args.checkpoint_interval,
    )
elif args.pretrain == 'supervised':
    data_reader_file = os.path.join(current_dir, 'data_reader_imagenet.prototext')
    model1, reader1, opt1 = classifier.setup(
        data_reader_file=data_reader_file,
        name='supervised',
        num_labels=1000,
        mini_batch_size=512,
        num_epochs=args.pretrain_epochs,
        learning_rate=0.1,
        warmup_epochs=5,
        learning_rate_drop_interval=30,
        learning_rate_drop_factor=0.1,
        checkpoint_interval=args.checkpoint_interval,
    )
else:
    raise Exception(f'"{args.pretrain}" is an invalid pretraining model')
model1.random_seed = random.getrandbits(32)

# Fine-tuning model
data_reader_file = os.path.join(current_dir, 'data_reader_cub.prototext')
model2, reader2, opt2 = classifier.setup(
    data_reader_file=data_reader_file,
    name='finetune',
    num_labels=200,
    mini_batch_size=128,
    num_epochs=500,
    learning_rate=0.1,
    warmup_epochs=50,
    learning_rate_drop_interval=50,
    learning_rate_drop_factor=0.25,
)

# ==============================================
# Construct LBANN invocation
# ==============================================

# Initialize LBANN executable and command-line arguments
lbann_exe = os.path.realpath(lbann.lbann_exe())
lbann_exe = os.path.join(os.path.dirname(lbann_exe), 'lbann2')
lbann_command = [lbann_exe]

# Construct experiment directory
experiment_dir = util.make_experiment_dir(args.job_name)

# Export model prototext files
# Note: lbann2 driver doesn't have a command-line argument to get
# trainer.
file1 = os.path.join(experiment_dir, 'model1.prototext')
file2 = os.path.join(experiment_dir, 'model2.prototext')
lbann.proto.save_prototext(file1, model=model1, trainer=lbann.Trainer(mini_batch_size=512))
lbann.proto.save_prototext(file2, model=model2, trainer=lbann.Trainer(mini_batch_size=512))
lbann_command.append(f'--model={{{file1},{file2}}}')

# Export data reader prototext files
file1 = os.path.join(experiment_dir, 'reader1.prototext')
file2 = os.path.join(experiment_dir, 'reader2.prototext')
lbann.proto.save_prototext(file1, data_reader=reader1)
lbann.proto.save_prototext(file2, data_reader=reader2)
lbann_command.append(f'--reader={{{file1},{file2}}}')

# Export optimizer prototext files
file1 = os.path.join(experiment_dir, 'opt1.prototext')
file2 = os.path.join(experiment_dir, 'opt2.prototext')
lbann.proto.save_prototext(file1, optimizer=opt1)
lbann.proto.save_prototext(file2, optimizer=opt2)
lbann_command.append(f'--optimizer={{{file1},{file2}}}')

# ==============================================
# Launch experiment
# ==============================================

# Construct batch script
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
script = lbann.contrib.launcher.make_batch_script(
    work_dir=experiment_dir,
    job_name=args.job_name,
    **kwargs,
)
script.add_parallel_command(lbann_command)

# Launch LBANN
if args.batch_job:
    script.submit()
else:
    script.run()
