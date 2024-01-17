from abc import ABC, abstractmethod
import os
import inspect
import pickle
import lbann


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
