from tqdm import trange
from multiprocessing import Pool
import numpy as np
import pickle


class Processor:

    def __init__(self, total_threads: int):
        self.threads = total_threads

    def __call__(self, tid: int):
        import ..thepile as dataset
        num_samples = dataset.num_val_samples()
        filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized/val.bin'

        result = [None] * num_samples
        for i in trange(num_samples):
            sample = np.array(dataset.get_val_sample(i), dtype=np.int32)
            result[i] = sample

        print('Saving...')

        with open(filename, 'wb') as fp:
            pickle.dump(result, fp)

        print('Done')


if __name__ == '__main__':
    threads = 1
    with Pool(threads) as pool:
        pool.map(Processor(threads), range(threads))
