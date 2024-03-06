from abc import ABC, abstractmethod
import os
import inspect
import pickle
import lbann
from multiprocessing import Process, Queue, Value, Condition
import numpy as np


class Sample:
    def __init__(self, sample=None, label=None, response=None, label_reconstruction=None):
        if sample is not None:
            self.sample = sample
        if label is not None:
            self.label = label
        if response is not None:
            self.response = response
        if label_reconstruction is not None:
            self.label_reconstruction = label_reconstruction


class SampleDims:
    def __init__(self, sample=None, label=None, response=None, label_reconstruction=None):
        if sample is not None:
            self.sample = sample
        if label is not None:
            self.label = label
        if response is not None:
            self.response = response
        if label_reconstruction is not None:
            self.label_reconstruction = label_reconstruction


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> Sample:
        pass

    @abstractmethod
    def get_sample_dims(self) -> SampleDims:
        pass


class DistConvDataset(Dataset):
    @property
    def rank(self):
        if not hasattr(self, '_rank'):
            self._rank = 0
        return self._rank
    
    @rank.setter
    def rank(self, rank):
        self._rank = rank
    
    @property
    def num_io_partitions(self):
        if not hasattr(self, '_num_io_partitions'):
            self._num_io_partitions = 1
        return self._num_io_partitions

    @num_io_partitions.setter
    def num_io_partitions(self, num_io_partitions):
        self._num_io_partitions = num_io_partitions


class DataReader:
    def __init__(self, dataset, num_procs, prefetch_factor, dtype):
        self.dataset = dataset
        self.num_procs = num_procs
        self.prefetch_factor = prefetch_factor
        self.dtype = dtype
        self.queue = Queue()
        # Note: one sample is always sitting in worker memory, hence the
        # prefetch_factor - 1 for the queue size below
        self.rqueue = Queue(maxsize=max(1, (self.prefetch_factor - 1) * self.num_procs))
        self.tasks_queued = 0
        self.next_task = Value('i', 0)
        self.condition = Condition()

        self.procs = [Process(target=self.worker) for _ in range(self.num_procs)]
        for p in self.procs:
            p.start()

    def terminate(self):
        for p in self.procs:
            p.terminate()
    
    def worker(self):
        import signal
        for sig in range(signal.NSIG):
            try:
                signal.signal(sig, signal.SIG_DFL)
                pass
            except: pass

        while True:
            task_id, ind = self.queue.get()

            sample = self.dataset[ind]
            
            with self.condition:
                self.condition.wait_for(lambda: task_id == self.next_task.value)
                
                self.rqueue.put(sample)
                self.next_task.value += 1

                self.condition.notify_all()

    def queue_epoch(self, inds):
        for ind in inds:
            self.queue.put((self.tasks_queued, ind))
            self.tasks_queued += 1

    def get_batch(self, batch_size):
        samples = [self.rqueue.get() for _ in range(batch_size)]

        batch = {}

        if hasattr(samples[0], 'sample'):
            batch['sample'] = np.ascontiguousarray([s.sample for s in samples], dtype=self.dtype)
            batch['sample_ptr'] = batch['sample'].ctypes.data

        if hasattr(samples[0], 'label'):
            batch['label'] = np.ascontiguousarray([s.label for s in samples], dtype=self.dtype)
            batch['label_ptr'] = batch['label'].ctypes.data

        if hasattr(samples[0], 'response'):
            batch['response'] = np.ascontiguousarray([s.response for s in samples], dtype=self.dtype)
            batch['response_ptr'] = batch['response'].ctypes.data

        if hasattr(samples[0], 'label_reconstruction'):
            batch['label_reconstruction'] = np.ascontiguousarray([s.label_reconstruction for s in samples], dtype=self.dtype)
            batch['label_reconstruction_ptr'] = batch['label_reconstruction'].ctypes.data
        
        return batch
    

def construct_python_dataset_reader(dataset,
                                    dataset_path=None,
                                    role='train',
                                    shuffle=True,
                                    fraction_of_data_to_use=1.0,
                                    validation_fraction=0.,
                                    load_module=True):
    if dataset_path is None:
        main_file = inspect.stack()[-1].filename
        dataset_path = os.path.join(os.path.dirname(main_file), f'{role}_dataset.pkl')
    
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)

    module_dir = None
    if load_module:
        module_dir = os.path.dirname(inspect.getsourcefile(type(dataset)))
    
    reader = lbann.reader_pb2.Reader(
        name='python_dataset',
        role=role,
        shuffle=shuffle,
        fraction_of_data_to_use=fraction_of_data_to_use,
        validation_fraction=validation_fraction,
        python_dataset=lbann.reader_pb2.PythonDatasetReader(
            dataset_path=dataset_path,
            module_dir=module_dir
        )
    )

    return reader
