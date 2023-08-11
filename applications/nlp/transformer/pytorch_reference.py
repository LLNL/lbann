import argparse
import torch
import random
import tqdm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mini-batch-size',
                        action='store',
                        default=256,
                        type=int,
                        help='mini-batch size (default: 256)',
                        metavar='NUM')
    parser.add_argument('--num-epochs',
                        action='store',
                        default=20,
                        type=int,
                        help='number of epochs (default: 20)',
                        metavar='NUM')
    parser.add_argument(
        '--num-attention-heads',
        action='store',
        default=8,
        type=int,
        help='number of parallel attention layers (default: 8)',
        metavar='NUM')
    parser.add_argument('--embed-dim',
                        action='store',
                        default=512,
                        type=int,
                        help='embedding space dimensions (default: 512)',
                        metavar='NUM')

    parser.add_argument(
        '--num-layers',
        action='store',
        default=6,
        type=int,
        help='Number of encoder and decoder layers (default: 6)',
        metavar='NUM')

    parser.add_argument('--synthetic',
                        action='store_true',
                        help='Use synthetic data')

    parser.add_argument('--dataset-fraction',
                        action='store',
                        default=1.0,
                        type=float,
                        help='Fraction of dataset to use',
                        metavar='NUM')

    # PyTorch specific arguments
    parser.add_argument('--compile',
                        action='store_true',
                        default=False,
                        help='Use torch.compile')

    args = parser.parse_args()

    # Setup dataset
    if args.synthetic:
        import dataset_synthetic as dataset
    else:
        import dataset

    indices = list(range(dataset.num_train_samples()))
    num_samples = int(len(indices) * args.dataset_fraction)
    b = args.mini_batch_size

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
            self.embeddings = torch.nn.Embedding(dataset.vocab_size(),
                                                 args.embed_dim,
                                                 dataset.pad_index)
            self.linear = torch.nn.Linear(args.embed_dim, dataset.vocab_size())

        def forward(self, samples):
            src = samples[:, :dataset.sequence_length]
            tgt = samples[:, dataset.sequence_length:-1]
            esrc = self.embeddings(src)
            etgt = self.embeddings(tgt)

            output = self.model(esrc, etgt)
            result = self.linear(output)
            return result

    class TransformerCriterion(torch.nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.smax = torch.nn.Softmax(-1)
            self.ce = torch.nn.CrossEntropyLoss()
            self.pad = dataset.pad_index
            self.vocab_size = dataset.vocab_size()

        def forward(self, output, labels):
            s = self.smax(output)
            s = s.view(-1, self.vocab_size)
            labels = labels.view(-1)

            # Ignore padding by setting values to a size that will not contribute
            s = s[labels != self.pad]
            labels = labels[labels != self.pad]

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
        for idx in tqdm.trange(num_samples // b):
            # Stack minibatch
            samples = torch.tensor(
                [dataset.get_train_sample(indices[idx + i]) for i in range(b)],
                dtype=torch.int32).cuda()

            labels = samples[:, dataset.sequence_length + 1:].long()

            # Call model
            opt.zero_grad()
            preds = model(samples)
            loss = criterion(preds, labels)
            loss.backward()
            opt.step()


if __name__ == '__main__':
    main()
