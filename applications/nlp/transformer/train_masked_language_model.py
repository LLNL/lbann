"""
Driver script for LBANN pre-training a masked language modeling task.
It receives a data loader script and newline-separated strings, or a dataset
name (see ``pretrain_gpt.py`` for example), and trains the transformer by
randomly masking out tokens at ``--mlm-fraction`` probability.
"""
import argparse
from dataclasses import dataclass
import math
import os
import os.path
import sys

import lbann
import lbann.contrib.args
from lbann.launcher.batch_script import BatchScript
import lbann.util.amp

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import dataset_utils
import modeling
import parallelism
import trainer

# In this sample, we are training on molecular data
from lbann.contrib.data.molecule_dataset import ChemTextDataset, ChemTokenType


@dataclass
class GPTConfig:
    """
    A simple size configuration entry for GPT-3 models.
    """
    layers: int
    model_dim: int
    num_heads: int
    head_dim: int
    lr: float


SIZES = {
    'small': GPTConfig(12, 768, 12, 64, 6e-4),
    'medium': GPTConfig(24, 1024, 16, 64, 3e-4),
    'large': GPTConfig(24, 1536, 16, 96, 2.5e-4),
    'xl': GPTConfig(24, 2048, 24, 128, 2e-4),
    '2.7b': GPTConfig(32, 2560, 32, 80, 1.6e-4),
    '6.7b': GPTConfig(32, 4096, 32, 128, 1.2e-4),
    '13b': GPTConfig(40, 5120, 40, 128, 1e-4),
    'gpt3': GPTConfig(96, 12288, 96, 128, 6e-5),
}


def main():
    # Setup command line options
    parser = argparse.ArgumentParser()
    lbann.contrib.args.add_scheduler_arguments(parser, 'lbann_mlm')
    lbann.contrib.args.add_profiling_arguments(parser)
    lbann.contrib.args.add_training_arguments(parser,
                                              default_minibatch_size=32,
                                              default_epochs=1)
    lbann.contrib.args.add_amp_arguments(parser)
    parallelism.add_transformer_parallelism_arguments(parser)
    trainer.add_training_arguments(parser)
    dataset_utils.add_dataset_arguments(parser, default=None)

    parser.add_argument('--optimizer',
                        type=str,
                        default='adamw',
                        choices=['adam', 'adamw'],
                        help='Stochastic optimizer used in training')

    # Model parameters
    parser.add_argument('--model-type',
                        default='small',
                        type=str,
                        help='The type of GPT model to use (default: small)',
                        choices=SIZES.keys())
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout ratio in transformer model. 0 disables (default: 0.0)")
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.1,
        help="Dropout ratio after input encoding. 0 disables (default: 0.1)")
    parser.add_argument(
        "--attn-dropout",
        type=float,
        default=0.0,
        help="Dropout ratio after multi-head attention (default: 0.0)")
    parser.add_argument("--grad-clip",
                        type=float,
                        default=0.0,
                        help="Clip global gradient norm (default: 0.0)")
    parser.add_argument(
        "--fractional-schedule",
        action='store_true',
        default=False,
        help="Use dataset fraction to determine hyperparameter schedule"
        " (default: False)")
    parser.add_argument(
        '--positional-encoding',
        type=str,
        default='learned',
        help='The type of positional encoding to use '
        '(default: learned)',
        choices=[s.name.lower() for s in modeling.InputEncoding])

    # Masking properties
    parser.add_argument(
        "--encoder",
        action='store_true',
        default=False,
        help="Use encoders instead of decoders (disables causal attention)"
        " (default: False)")
    parser.add_argument(
        "--attn-mask",
        action='store_true',
        default=False,
        help="Use custom attention mask (incompatible with --encoder)"
        " (default: False)")
    parser.add_argument("--mlm-fraction",
                        type=float,
                        default=0.15,
                        help="Fraction of tokens to mask (default: 0.15)")

    # Dataset properties
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Sequence length in dataset to use (default: 256)")
    parser.add_argument("--vocab-file",
                        type=str,
                        help="Vocabulary text file to use")
    parser.add_argument(
        "--train-set",
        action="append",
        help="Path to training dataset file or files (newline-separated text file)")
    parser.add_argument("--data-format",
                        type=str,
                        help="Dataset format",
                        default='smiles',
                        choices=[s.name.lower() for s in ChemTokenType])
    parser.add_argument(
        "--token-format",
        type=str,
        help="Tokenized data format (specify if different than `--data-format`",
        default=None,
        choices=[s.name.lower() for s in ChemTokenType])
    parser.add_argument(
        "--val-set",
        action="append",
        help="Path to validation dataset file or files (newline-separated text file)")
    parser.add_argument(
        "--dataset-prefetch-factor",
        type=int,
        default=1,
        help="Prefetching factor for training dataset (default: 1)")

    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if args.job_name == 'lbann_mlm':
        args.job_name = f'lbann_mlm_{args.model_type}'

    if args.dataset and (args.train_set or args.val_set):
        raise ValueError('Either --dataset or --train-set must be passed.')
    if not args.dataset and (not args.train_set or not args.vocab_file):
        raise ValueError(
            'The following arguments are required: --vocab-file, --train-set')
    if args.attn_mask and args.encoder:
        raise ValueError('--attn-mask is incompatible with --encoder')

    # Load dataset
    val_dataset = None
    if args.dataset:
        dataset = dataset_utils.load_dataset(args.dataset)
    else:
        input_format = ChemTokenType[args.data_format.upper()]
        if args.token_format is not None:
            token_format = ChemTokenType[args.token_format.upper()]
        else:
            token_format = input_format

        dataset = ChemTextDataset(args.train_set,
                                  args.vocab_file,
                                  args.sequence_length,
                                  tokenizer_type=token_format,
                                  input_type=input_format,
                                  mlm_probability=args.mlm_fraction,
                                  attn_mask=args.attn_mask)
        if args.val_set:
            val_dataset = ChemTextDataset(args.val_set,
                                          args.vocab_file,
                                          args.sequence_length,
                                          tokenizer_type=token_format,
                                          input_type=input_format,
                                          mlm_probability=args.mlm_fraction,
                                          attn_mask=args.attn_mask)

    # Construct model
    chosen_config = SIZES[args.model_type]
    model: lbann.Model = modeling.create_masked_language_modeling_transformer(
        dataset,
        embed_dim=chosen_config.head_dim * chosen_config.num_heads,
        num_encoders=0 if not args.encoder else chosen_config.layers,
        num_decoders=0 if args.encoder else chosen_config.layers,
        num_heads=chosen_config.num_heads,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        attn_dropout=args.attn_dropout,
        num_epochs=args.num_epochs,
        args=args,
    )
    lbann.util.amp.enable_amp(model, args)

    # Construct trainer

    # Training schedule
    # GPT-3 paper used 300 billion tokens in total
    lr_decay_ratio = 260 / 300  # 260 billion tokens used for cosine decay
    warmup_ratio = 375 / 300000  # 375 million tokens for warmup
    # tokens_per_step = dataset.sequence_length * args.mini_batch_size
    sched_mult = args.dataset_fraction if args.fractional_schedule else 1.0
    total_steps = math.ceil(sched_mult * dataset.num_train_samples() /
                            args.mini_batch_size)

    train_script: BatchScript = trainer.construct_training_task(
        model,
        args,
        learning_rate=chosen_config.lr,
        # Training parameters from paper
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        clip_gradient=args.grad_clip,
        lr_decay='cosine',
        lr_decay_steps=int(lr_decay_ratio * total_steps),
        end_learning_rate=chosen_config.lr / 10,
        warmup_steps=int(warmup_ratio * total_steps),
        adamw_decay=0.1,
        dataset=args.dataset or dataset,
        validation_dataset=val_dataset,
    )

    # Run trainer
    if not args.setup_only:
        train_script.run(overwrite=True)
    else:
        train_script.write(overwrite=True)


if __name__ == '__main__':
    main()
