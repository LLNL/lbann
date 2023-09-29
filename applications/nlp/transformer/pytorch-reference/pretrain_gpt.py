"""
Pre-train a GPT-3-like model with a causal language modeling target.
Requires HuggingFace Transformers. The default dataset (The Pile) 
also requires HuggingFace Datasets and Tokenizers.
"""
import argparse
from dataclasses import dataclass
import torch
import random
import tqdm
import numpy as np
import sys
import os

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

# Import local utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataset_utils


@dataclass
class GPTConfig:
    layers: int
    model_dim: int
    num_heads: int
    head_dim: int


sizes = {
    'small': GPTConfig(12, 768, 12, 64),
    'medium': GPTConfig(24, 1024, 16, 64),
    'large': GPTConfig(24, 1536, 16, 96),
    'xl': GPTConfig(24, 2048, 24, 128),
    '2.7b': GPTConfig(32, 2560, 32, 80),
    '6.7b': GPTConfig(32, 4096, 32, 128),
    '13b': GPTConfig(40, 5140, 40, 128),
    'gpt3': GPTConfig(96, 12288, 96, 128),
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mini-batch-size',
                        action='store',
                        default=32,
                        type=int,
                        help='Mini-batch size (default: 32)',
                        metavar='NUM')

    parser.add_argument('--num-epochs',
                        action='store',
                        default=20,
                        type=int,
                        help='Number of epochs (default: 20)',
                        metavar='NUM')

    parser.add_argument('--model-type',
                        default='small',
                        type=str,
                        help='The type of GPT model to use (default: small)',
                        choices=sizes.keys())

    parser.add_argument('--max-sequence-length',
                        action='store',
                        default=2048,
                        type=int,
                        help='Maximal sequence length (default: 2048)',
                        metavar='NUM')

    parser.add_argument(
        '--train-sequence-length',
        action='store',
        default=0,
        type=int,
        help='Sequence length for training. 0 to keep as-is (default: 0)',
        metavar='NUM')

    dataset_utils.add_dataset_arguments(parser, default='thepile')

    parser.add_argument('--compile',
                        action='store_true',
                        default=False,
                        help='Use torch.compile')

    args = parser.parse_args()

    # Load the dataset
    dataset = dataset_utils.load_dataset(args.dataset)

    if args.train_sequence_length > 0:
        print('Setting sequence length to', args.train_sequence_length)
        dataset.sequence_length = args.train_sequence_length

    num_samples = int(dataset.num_train_samples() * args.dataset_fraction)
    b = args.mini_batch_size
    config = sizes[args.model_type]

    print('Constructing model...')
    model = GPT2LMHeadModel(
        GPT2Config(dataset.vocab_size(), args.max_sequence_length,
                   config.model_dim, config.layers, config.num_heads))

    model_size = sum(t.numel() for t in model.parameters())
    print(f'Model size: {model_size*1e-6:.1f}M parameters')
    model = model.cuda()

    if args.compile:
        print('Using torch.compile')
        model = torch.compile(model)

    train_loop(model, dataset, args.num_epochs, num_samples, b)

def train_loop(model, dataset, num_epochs, num_samples, b):
    # Loop through epochs
    opt = torch.optim.AdamW(model.parameters())
    for epoch in range(num_epochs):
        print('Epoch', epoch)
        indices = np.random.permutation(num_samples)
        model.train()
        progress = tqdm.tqdm(range(num_samples // b), total=num_samples // b)
        for idx in progress:
            # Stack minibatch
            samples = torch.tensor(np.array([
                dataset.get_train_sample(int(indices[idx + i]))
                for i in range(b)
            ]),
                                   dtype=torch.int64).cuda()

            # Call model
            opt.zero_grad()

            # Use causal LM
            preds = model(samples, labels=samples)
            if preds.loss is not None:
                max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

                progress.set_description(f'Loss: {preds.loss.item():.4f}. '
                                         f'Mem usage: {max_mem:.2f} MB')
                preds.loss.backward()
                opt.step()


if __name__ == '__main__':
    main()
