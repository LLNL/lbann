import argparse
import data_utils


# Command-line arguments
def add_transformer_architecture_arguments(args: argparse.Namespace):
    """
    Adds the command line arguments to specify transformer architecture model
    parameters. This is only relevant for the encoder-decoder transformer model.
    """
    args.add_argument('--num-attention-heads',
                      action='store',
                      default=8,
                      type=int,
                      help='number of parallel attention layers (default: 8)',
                      metavar='NUM')
    args.add_argument('--embed-dim',
                      action='store',
                      default=512,
                      type=int,
                      help='embedding space dimension (default: 512)',
                      metavar='NUM')
    args.add_argument('--feedforward-dim',
                      action='store',
                      default=0,
                      type=int,
                      help='feedforward network dimension. If zero, set to be '
                      '4 times the embedding dimension (default: 0)',
                      metavar='NUM')
    args.add_argument('--num-layers',
                      action='store',
                      default=6,
                      type=int,
                      help='Number of encoder and decoder layers (default: 6)',
                      metavar='NUM')


def add_dataset_arguments(args: argparse.Namespace, default: str):
    """
    Adds dataset-related arguments to an existing argparse object.
    """
    args.add_argument('--dataset',
                      type=str,
                      default=default,
                      help=f'Which dataset to use (default: {default})',
                      choices=data_utils.available_datasets())
    args.add_argument('--dataset-fraction',
                      action='store',
                      default=1.0,
                      type=float,
                      help='Fraction of dataset to use (default: 1.0)',
                      metavar='NUM')
