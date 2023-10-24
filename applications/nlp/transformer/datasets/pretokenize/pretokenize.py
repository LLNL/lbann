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
        import ..thepile as dataset
        num_samples = dataset.num_train_samples()
        np.random.seed(20231023)
        indices = np.random.permutation(num_samples)
        local_samples = num_samples // self.threads
        offset = tid * local_samples
        # Add remainder
        if tid == self.threads - 1:
            local_samples += num_samples % self.threads
        section = indices[offset:offset + local_samples]
        filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized/train-pretokenized-{tid:02d}-of-{self.threads}.bin'

        # Create file
        if not os.path.isfile(filename):
            Path(filename).touch()

        sz = os.path.getsize(filename)
        assert sz % (dataset.sequence_length * 4) == 0
        sequences_processed = sz // (dataset.sequence_length * 4)
        print(tid, ': Size in bytes:', sz, '. Sequences processed:',
              sequences_processed)

        with open(filename, 'ab') as fp:
            for i in trange(sequences_processed,
                            section.shape[0],
                            desc=f'Thread {tid}'):
                sample = np.array(dataset.get_train_sample(int(section[i])),
                                  dtype=np.int32)
                sample.tofile(fp)

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
