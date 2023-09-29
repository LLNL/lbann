import argparse
import lbann
import lbann.contrib.args
import lbann.contrib.launcher
from gan_model import build_model
from mnist_dataset import make_data_reader

desc = ('Train GAN on MNIST data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_mnist_gan')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
args = parser.parse_args()

trainer = lbann.Trainer(args.mini_batch_size)
model = build_model(args.num_epochs)
data_reader = make_data_reader()
opt = lbann.Adam(learn_rate=1e-4, beta1=0., beta2=0.99, eps=1e-8)

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           **kwargs)
