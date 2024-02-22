from tqdm import trange
from multiprocessing import Pool
import numpy as np
import pickle


class Processor:

    def __init__(self, total_threads: int):
        self.threads = total_threads

    def __call__(self, tid: int):
        import thepile as dataset
        num_samples = dataset.num_val_samples()
        filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized-varlen/val.bin'
        len_filename = f'/p/vast1/data/datasets/the-pile-huggingface/pretokenized-varlen/val-seqlen.bin'

        with open(filename, 'ab') as fp:
            with open(len_filename, 'ab') as slfp:
                for i in trange(num_samples):
                    text = dataset.dataset_val[i]['text']
                    tokenized = dataset.tokenize(text)
                    sample = np.array(tokenized, dtype=np.uint16)
                    sample_len = np.array([len(sample)], dtype=np.uint32)
                    sample.tofile(fp)
                    sample_len.tofile(slfp)

        print('Done')


if __name__ == '__main__':
    threads = 1
    with Pool(threads) as pool:
        pool.map(Processor(threads), range(threads))
