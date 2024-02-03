from tqdm import trange
from multiprocessing import Pool
import numpy as np
import os
import argparse
from pathlib import Path


class Processor:

    def __init__(self, total_threads: int):
        self.threads = total_threads

    def __call__(self, tid: int):
        import thepile as dataset
        num_samples = dataset.num_train_samples()
        np.random.seed(20231023)
        indices = np.random.permutation(num_samples)
        local_samples = num_samples // self.threads
        offset = tid * local_samples
        # Add remainder
        if tid == self.threads - 1:
            local_samples += num_samples % self.threads
        section = indices[offset:offset + local_samples]
        filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized-varlen/train-pretokenized-{tid:02d}-of-{self.threads}.bin'
        len_filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized-varlen/train-seqlen-{tid:02d}-of-{self.threads}.bin'

        # Create file
        if not os.path.isfile(filename):
            Path(filename).touch()
        if not os.path.isfile(len_filename):
            Path(len_filename).touch()

        sz = os.path.getsize(len_filename)
        assert sz % 4 == 0
        sequences_processed = sz // 4
        print(tid, ': Size in bytes:', sz, '. Sequences processed:',
              sequences_processed)

        with open(filename, 'ab') as fp:
            with open(len_filename, 'ab') as slfp:
                for i in trange(sequences_processed,
                                section.shape[0],
                                desc=f'Thread {tid}'):
                    text = dataset.dataset_train[int(section[i])]['text']
                    sample = dataset.tokenize(text)
                    sample = np.array(sample, dtype=np.uint16)
                    sample.tofile(fp)
                    sample_len = np.array([len(sample)], dtype=np.uint32)
                    sample_len.tofile(slfp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-j',
                        action='store',
                        default=0,
                        type=int,
                        help='Threads (default 0 = number of cores)')
    parser.add_argument('-t',
                        action='store',
                        default=0,
                        type=int,
                        help='Total Chunks (default 0 = number of threads)')
    parser.add_argument('-o',
                        action='store',
                        default=0,
                        type=int,
                        help='Chunk offset (default 0)')
    args = parser.parse_args()

    threads = args.j or os.cpu_count()
    total_chunks = args.t or threads
    offset = args.o
    assert offset + threads <= total_chunks
    with Pool(threads) as pool:
        pool.map(Processor(total_chunks), range(offset, offset + threads))
