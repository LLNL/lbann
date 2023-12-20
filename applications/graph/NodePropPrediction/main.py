from dataset_wrapper import DATASET_CONFIG
import lbann
import lbann.contrib.launcher
import lbann.contrib.args

import argparse

desc = " Training a Graph Convolutional Model using LBANN"
parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser, "GNN")
lbann.contrib.args.add_optimizer_arguments(parser)

parser.add_argument(
    "--num-epochs",
    action="store",
    default=100,
    type=int,
    help="number of epochs (deafult: 100)",
    metavar="NUM",
)

parser.add_argument(
    "--model",
    action="store",
    default="GCN",
    type=str,
    help="The type of model to use",
    metavar="NAME",
)

parser.add_argument(
    "--dataset",
    action="store",
    default="ARXIV",
    type=str,
    help="The dataset to use",
    metavar="NAME",
)

parser.add_argument(
    "--latent-dim",
    action="store",
    default=16,
    type=int,
    help="The latent dimension of the model",
    metavar="NUM",
)

parser.add_argument(
    "--num-layers",
    action="store",
    default=3,
    type=int,
    help="The number of layers in the model",
    metavar="NUM",
)


SUPPORTED_MODELS = ["GCN", "GAT"]
SUPPORTED_DATASETS = ["ARXIV", "PRODUCTS", "MAG240M"]


def main():
    args = parser.parse_args()

    kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

    num_epochs = args.num_epochs
    mini_batch_size = 1
    job_name = args.job_name
    model_arch = args.model
    dataset = args.dataset

    if model_arch not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model {model_arch} not supported. Supported models are {SUPPORTED_MODELS}"
        )

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset {dataset} not supported. Supported datasets are {SUPPORTED_DATASETS}"
        )
    dataset_config = DATASET_CONFIG[dataset]
    num_nodes = dataset_config["num_nodes"]
    num_edges = dataset_config["num_edges"]
    input_features = dataset_config["input_features"]


    optimizer = lbann.SGD(learn_rate=0.01, momentum=0.0, eps=1e-8)
    lbann.contrib.launcher.run(trainer, model, data_reader, opt, **kwargs)
