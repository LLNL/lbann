"""
Contains an LBANN dataset reader for molecular text datasets in various
input formats.
"""

import numpy as np
import os
from typing import List, Union

from lbann.contrib.data.chem_tokenizers import TOKENIZERS, ChemTokenType
from lbann.util.data import Dataset as LBANNDataset, Sample, SampleDims
from torch.utils.data import Dataset


def concat(*args):
    return np.concatenate(tuple(a.flat for a in args))


class TextDataset(Dataset):
    """
    Simple PyTorch text dataset. Adapted from the now-deprecated 
    ``LineByLineTextDataset`` in HuggingFace Transformers.
    """

    def __init__(self, file_or_files: Union[str, List[str]]):
        if isinstance(file_or_files, str):
            file_or_files = [file_or_files]

        self.lines: List[str] = []
        for file in file_or_files:
            with open(file, 'r') as fp:
                self.lines.extend(fp.readlines())

        # Clean empty lines
        self.lines = [line.strip() for line in self.lines if line]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i: int) -> str:
        return self.lines[i]


class TextDatasetWithOffsets(Dataset):
    """
    PyTorch text dataset with offset files. Not loaded to memory but read
    on demand.
    """

    def __init__(self, file_or_files: Union[str, List[str]]):
        if isinstance(file_or_files, str):
            file_or_files = [file_or_files]
        self.files = file_or_files

        self.offsets = [
            np.memmap(f + '.offsets', dtype=np.uint64) for f in file_or_files
        ]
        self.samples = [o.shape[0] for o in self.offsets]
        self.cs = np.cumsum(np.array(self.samples, dtype=np.uint64), dtype=np.uint64)
        self.total_samples = sum(self.samples)

        # Clean memmapped files so that the object can be pickled
        self.offsets = None
        #self.files = [np.memmap(f, dtype=np.uint8) for f in file_or_files]

    def _lazy_reload(self):
        if self.offsets is not None:
            return

        self.offsets = [
            np.memmap(f + '.offsets', dtype=np.uint64) for f in self.files
        ]
        self.files = [np.memmap(f, dtype=np.uint8) for f in self.files]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, i: int) -> str:
        self._lazy_reload()

        # Find file
        f = self.cs.searchsorted(i, side='right')
        local_i = i - (self.cs[f - 1].item() if f > 0 else 0)

        # Find offset
        if local_i == 0:
            soff = 0
        else:
            soff = self.offsets[f][local_i - 1].item() + 1
        eoff = self.offsets[f][local_i].item()

        # Return string
        return self.files[f][soff:eoff].tobytes().decode('utf-8')


class ChemTextDataset(LBANNDataset):
    """
    A molecular dataset reader supporting different atom formats and tokenizers.
    """

    def __init__(self,
                 fname: Union[str, List[str]],
                 vocab: str,
                 seqlen: int,
                 tokenizer_type: ChemTokenType,
                 mlm_probability: float = 0.15,
                 attn_mask: bool = False,
                 input_type: ChemTokenType = None):
        """
        Initializes a molecular text dataset

        :param fname: File name or list of file names to read.
        :param vocab: File path containing a newline-separated list of
                      vocabulary entries.
        :param seqlen: A sequence length to align (pad/trim) all sequences to.
        :param tokenizer_type: The tokenizer type to use when training on
                               molecules.
        :param mlm_probability: Probability to mask out a given token.
        :param attn_mask: If True, passes an attention probability mask to the
                          model.
        :param input_type: Specifies a separate input type when reading the
                           molecules from dataset file(s). If None, uses the
                           same representation as ``tokenizer_type``.
        """
        super().__init__()
        self.sequence_length = seqlen
        self.tokenizer = TOKENIZERS[tokenizer_type](vocab)

        # Make decoder based on input type
        input_type = input_type or tokenizer_type
        if input_type == ChemTokenType.SMILES:
            self.decode = self.tokenizer.from_smiles
        elif input_type == ChemTokenType.SELFIES:
            self.decode = self.tokenizer.from_selfies
        elif input_type == ChemTokenType.AIS:
            self.decode = self.tokenizer.from_ais
        else:
            raise ValueError(f'Unrecognized input type {input_type}')

        if os.path.exists(fname[0] + '.offsets'):
            print('Using fast text dataset with offsets')
            self.dataset = TextDatasetWithOffsets(fname)
        else:
            self.dataset = TextDataset(fname)

        self._vocab_size = len(self.tokenizer)
        self.mlm_prob = mlm_probability
        self.pad_index = self.tokenizer.pad_token_id
        self.mask_index = self.tokenizer.mask_token_id
        self.attn_mask = attn_mask

    def vocab_size(self) -> int:
        return self._vocab_size

    def __len__(self) -> int:
        return len(self.dataset)

    def num_train_samples(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.decode(self.dataset[index])
        tokenized = self.tokenizer(sample)
        ids = np.array(tokenized['input_ids'], dtype=np.int32)

        if self.attn_mask:
            samp = Sample(
                concat(self.trim_and_pad(ids, True), self.make_mask(),
                       self.make_attn_mask()))
        else:
            samp = Sample(
                concat(self.trim_and_pad(ids, True), self.make_mask()))
        return samp

    def get_sample_dims(self):
        amask = 0 if not self.attn_mask else (self.sequence_length *
                                              self.sequence_length)
        return SampleDims(
            (self.sequence_length + self.sequence_length + amask, ))

    def make_mask(self, random: bool = True) -> np.ndarray:
        # 0 = masked, 1 = not masked
        if random:
            return np.random.binomial(1,
                                      1 - self.mlm_prob,
                                      size=self.sequence_length)

        # All masked:
        #return np.full((self.sequence_length, ), 0)
        # Nothing masked:
        return np.full((self.sequence_length, ), 1)

    def make_attn_mask(self) -> np.ndarray:
        # Additive mask: uses -inf for masked, 0 for nonmasked
        return np.triu(
            np.full((self.sequence_length, self.sequence_length), -1e9),
            k=1,
        )

    def trim_and_pad(self, sample, random: bool):
        # Trim long sequences
        if len(sample) > self.sequence_length:
            if random:
                pos = np.random.rand()
                offset = (len(sample) - self.sequence_length + 1) * pos
                offset = int(np.floor(offset))
                sample = sample[offset:offset + self.sequence_length]
            else:
                sample = sample[0:self.sequence_length]

        # Left-pad short sequences
        if len(sample) < self.sequence_length:
            sample_pad = np.full(self.sequence_length,
                                 self.pad_index,
                                 dtype=np.int32)
            if len(sample) > 0:
                sample_pad[-len(sample):] = sample
            return sample_pad

        return sample


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('USAGE: dataloader_mlm.py <dataset file> <vocabulary file> '
              '<smiles/selfies/ais>')
        exit(1)

    dataset = ChemTextDataset(
        fname=[sys.argv[1]],
        vocab=sys.argv[2],
        seqlen=64,
        tokenizer_type=ChemTokenType[sys.argv[3].upper()])
    print('Dataset samples:', len(dataset))
    print('Dataset sample -1:')
    print(
        dataset.tokenizer.decode(dataset[-1].sample[:dataset.sequence_length]))
