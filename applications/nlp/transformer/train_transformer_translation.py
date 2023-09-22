"""
Trains a Transformer (Encoder-Decoder) neural network on a source-target
sequence task. A WMT-16 dataset reader is provided as a sample translation task.
"""
import argparse
import os
import os.path
import sys

import lbann
import lbann.contrib.args
from lbann.launcher.batch_script import BatchScript

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import dataset_utils
import modeling
import trainer


def main():
    # Setup command line options
    parser = argparse.ArgumentParser()
    lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_transformer')
    lbann.contrib.args.add_profiling_arguments(parser)
    lbann.contrib.args.add_training_arguments(parser)
    trainer.add_kfac_arguments(parser)

    modeling.add_transformer_architecture_arguments(parser)
    dataset_utils.add_dataset_arguments(parser, default='wmt16')

    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        choices=['adam', 'adamw'],
                        help='Stochastic optimizer used in training')

    # Model parameters
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout ratio in transformer model. 0 disables (default: 0.1)")
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        help="Dropout ratio after input encoding (default: 0.0 = disabled)")

    args = parser.parse_args()

    # Load dataset
    dataset = dataset_utils.load_dataset(args.dataset)

    # Construct model
    model: lbann.Model = modeling.create_encoder_decoder_transformer(
        dataset, args)

    # Construct trainer
    train_script: BatchScript = trainer.construct_training_task(model, args)

    # Run trainer
    train_script.run(overwrite=True)

    if args.checkpoint:
        print(
            'Training complete, to evaluate the translation with a BLEU score, '
            'run ``evaluate_translation_bleu.py`` with the model checkpoint path '
            'and the same arguments as this training run.')
    else:
        print(
            'Training complete, to evaluate the translation with a BLEU score, '
            'run this script with --checkpoint')


if __name__ == '__main__':
    main()
