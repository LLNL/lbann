import lbann
import lbann.contrib.launcher
import lbann.contrib.args
from config import HYPERPARAMETERS_CONFIG
from model import make_model, make_data_reader
import argparse


desc = " Training a MPNN Model using LBANN"
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'ChemProp')
lbann.contrib.args.add_optimizer_arguments(parser)

args = parser.parse_args()
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
job_name = args.job_name

model = make_model()
data_reader = make_data_reader()
optimizer = lbann.SGD(learn_rate=HYPERPARAMETERS_CONFIG["LR"])
trainer = lbann.Trainer(mini_batch_size=HYPERPARAMETERS_CONFIG["BATCH_SIZE"])

lbann.contrib.launcher.run(
    trainer, model, data_reader, optimizer, job_name=job_name, **kwargs
)
