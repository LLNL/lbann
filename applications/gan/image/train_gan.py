import argparse
import lbann
import lbann.contrib.args
import lbann.contrib.launcher
from gan_model import build_model
from image_dataset import make_data_reader

desc = ('Train GAN on image data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_image_gan')
parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help='mini-batch size (default: 32)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
parser.add_argument(
    '--data-path', action='store', required=True, type=str,
    help='data path (required)', metavar='NUM')
args = parser.parse_args()

trainer = lbann.Trainer(args.mini_batch_size)
model = build_model(args.num_epochs)
data_reader = make_data_reader()
opt = lbann.Adam(learn_rate=1e-4, beta1=0., beta2=0.99, eps=1e-8)

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           job_name=args.job_name,
                           environment={'GAN_DATA_PATH': args.data_path},
                           **kwargs)
