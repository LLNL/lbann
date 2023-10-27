from tqdm import trange
import numpy as np
import os
from pathlib import Path


def process(files=80):
    for fid in trange(files):
        filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized/train-pretokenized-{fid:02d}-of-{files}.bin'
        print(filename)
        sequence_length = 512

        if not os.path.isfile('train.bin'):
            Path(filename).touch()

        with open('train.bin', 'ab') as wfp:
            with open(filename, 'rb') as fp:
                ds = np.fromfile(fp,
                                 dtype=np.int32).reshape(-1, sequence_length)
            ds.astype(np.uint16).tofile(wfp)


if __name__ == '__main__':
    process()
