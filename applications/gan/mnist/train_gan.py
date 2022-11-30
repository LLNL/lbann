import argparse
import lbann
import lbann.contrib.args
import lbann.contrib.launcher
from gan_model import build_model
from mnist_dataset import make_data_reader

desc = ('Train GAN on MNIST data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_mnist_gan', type=str,
    help='scheduler job name (default: lbann_mnist_gan)')
args = parser.parse_args()

mini_batch_size = 128
trainer = lbann.Trainer(mini_batch_size)

num_epochs = 100
model = build_model(num_epochs)

data_reader = make_data_reader()

opt = lbann.Adam(learn_rate=1e-4, beta1=0., beta2=0.99, eps=1e-8)

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
