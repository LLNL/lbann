"""
Train a Transformer model for translation.
The default dataset (WMT-16) requires HuggingFace Datasets and Tokenizers.
"""
import argparse
import torch
import math
import numpy as np
import os
import random
import sys
import tqdm

# Import local utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataset_utils


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

    parser.add_argument('--num-attention-heads',
                        action='store',
                        default=8,
                        type=int,
                        help='Number of parallel attention heads (default: 8)',
                        metavar='NUM')

    parser.add_argument('--embed-dim',
                        action='store',
                        default=512,
                        type=int,
                        help='Embedding space dimensions (default: 512)',
                        metavar='NUM')

    parser.add_argument(
        '--num-layers',
        action='store',
        default=6,
        type=int,
        help='Number of encoder and decoder layers (default: 6)',
        metavar='NUM')

    parser.add_argument(
        '--train-sequence-length',
        action='store',
        default=0,
        type=int,
        help='Sequence length for training. 0 to keep as-is (default: 0)',
        metavar='NUM')

    dataset_utils.add_dataset_arguments(parser, default='wmt16')

    # PyTorch specific arguments
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

    indices = list(range(dataset.num_train_samples()))
    num_samples = int(len(indices) * args.dataset_fraction)
    b = args.mini_batch_size

    class PositionalEncoding(torch.nn.Module):
        """
        Positional encoding, adapted from
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        for batch_first operation.
        """

        def __init__(self,
                     d_model: int,
                     dropout: float = 0.0,
                     max_len: int = 50000):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Arguments:
                x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            """
            x = x + self.pe[:x.size(1)]
            return self.dropout(x)

    # Setup model
    class TransformerModel(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()

            self.model = torch.nn.Transformer(
                d_model=args.embed_dim,
                nhead=args.num_attention_heads,
                num_encoder_layers=args.num_layers,
                num_decoder_layers=args.num_layers,
                batch_first=True,
            )
            self.pe = PositionalEncoding(args.embed_dim, 0.0,
                                         dataset.sequence_length)
            self.embeddings = torch.nn.Embedding(dataset.vocab_size(),
                                                 args.embed_dim,
                                                 dataset.pad_index)
            self.linear = torch.nn.Linear(args.embed_dim, dataset.vocab_size())

        def forward(self, samples):
            src = samples[:, :dataset.sequence_length]
            tgt = samples[:, dataset.sequence_length:-1]
            esrc = self.pe(self.embeddings(src))
            etgt = self.pe(self.embeddings(tgt))

            output = self.model(esrc, etgt)
            result = self.linear(output)
            return result

    class TransformerCriterion(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.smax = torch.nn.Softmax(-1)
            self.ce = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_index)
            self.vocab_size = dataset.vocab_size()

        def forward(self, output, labels):
            s = self.smax(output)
            s = s.view(-1, self.vocab_size)
            labels = labels.view(-1)

            return self.ce(s, labels)

    model = TransformerModel().cuda()
    criterion = TransformerCriterion().cuda()

    if args.compile:
        print('Using torch.compile')
        model = torch.compile(model)

    # Loop through epochs
    opt = torch.optim.AdamW(model.parameters())
    for epoch in range(args.num_epochs):
        print('Epoch', epoch)
        random.shuffle(indices)
        model.train()
        progress = tqdm.tqdm(range(num_samples // b), total=num_samples // b)
        for idx in progress:
            # Stack minibatch
            samples = torch.tensor(np.array([
                dataset.get_train_sample(indices[idx + i]) for i in range(b)
            ]),
                                   dtype=torch.int32).cuda()

            labels = samples[:, dataset.sequence_length + 1:].long()

            # Call model
            opt.zero_grad()
            preds = model(samples)
            loss = criterion(preds, labels)
            loss.backward()
            opt.step()
            progress.set_description(f'Loss: {loss.item():.4f}')


if __name__ == '__main__':
    main()
