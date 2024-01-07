from abc import ABC, abstractmethod


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


SampleDims = Sample


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
