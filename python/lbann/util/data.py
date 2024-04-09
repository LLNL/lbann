from abc import ABC, abstractmethod
import os
import inspect
import pickle
import lbann
from multiprocessing import Pool
import numpy as np
from typing import Dict, List, Optional, Union
from numpy.typing import ArrayLike


class Sample:
    """
    Represents a single sample of data.
    """

    def __init__(
        self,
        sample: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        response: Optional[ArrayLike] = None,
    ) -> None:
        """
        Sample Constructor

        :param sample: Sample input field, defaults to None
        :type sample: Optional[ArrayLike], optional
        :param label: Label input field, defaults to None
        :type label: Optional[ArrayLike], optional
        :param response: Response input field, defaults to None
        :type response: Optional[ArrayLike], optional
        """
        if sample is not None:
            self.sample = sample
        if label is not None:
            self.label = label
        if response is not None:
            self.response = response


class SampleDims:
    """
    Describes the dimensions of the samples returned by a dataset.
    """

    def __init__(
        self,
        sample: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        response: Optional[ArrayLike] = None,
    ) -> None:
        """
        SampleDims Constructor

        :param sample: Sample dimensions, defaults to None
        :type sample: Optional[ArrayLike], optional
        :param label: Label dimensions, defaults to None
        :type label: Optional[ArrayLike], optional
        :param response: Response dimensions, defaults to None
        :type response: Optional[ArrayLike], optional
        """
        if sample is not None:
            self.sample = sample
        if label is not None:
            self.label = label
        if response is not None:
            self.response = response


class Dataset(ABC):
    """
    Abstract base class for datasets.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        :return: Number of samples in the dataset
        :rtype: int
        """

        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        """
        Load the sample specified by the index.

        :param index: Sample index
        :type index: int
        :return: Sample
        :rtype: Sample
        """
        pass

    @abstractmethod
    def get_sample_dims(self) -> SampleDims:
        """
        Return the dimensions of the samples for each data field.

        :return: Sample dimensions of each data field
        :rtype: SampleDims
        """
        pass


class DistConvDataset(Dataset):
    """
    Abstract base class for DistConv enabled datasets.
    """

    @property
    def rank(self) -> int:
        """
        Rank in the trainer of this process.

        :return: Rank in trainer, defaults to 0
        :rtype: int
        """
        if not hasattr(self, "_rank"):
            self._rank = 0
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        """
        Setter for rank.

        :param rank: Rank in trainer
        :type rank: int
        """
        self._rank = rank

    @property
    def num_io_partitions(self) -> int:
        """
        Number of DistConv IO partitions.

        :return: Number of DistConv IO partitions, defaults to 1
        :rtype: int
        """
        if not hasattr(self, "_num_io_partitions"):
            self._num_io_partitions = 1
        return self._num_io_partitions

    @num_io_partitions.setter
    def num_io_partitions(self, num_io_partitions: int) -> None:
        """
        Setter for num_io_partitions.

        :param num_io_partitions: Number of DistConv IO partitions
        :type num_io_partitions: int
        """
        self._num_io_partitions = num_io_partitions


class DataReader:
    """
    Helper class used by LBANN to control worker processes and handle sample/batch loading.
    """

    def __init__(
        self, dataset: Dataset, num_procs: int, prefetch_factor: int, dtype: str
    ) -> None:
        """
        DataReader Constructor

        Launches worker processes during initialization.

        :param dataset: Dataset
        :type dataset: Dataset
        :param num_procs: Number of processes to be used in data loading
        :type num_procs: int
        :param prefetch_factor: Number of samples to prefetch per worker
        :type prefetch_factor: int
        :param dtype: Type of the batches to be returned
        :type dtype: str
        """
        self.dataset = dataset
        self.num_procs = num_procs
        self.prefetch_factor = prefetch_factor
        self.dtype = dtype
        self.sample_dims = dataset.get_sample_dims()
        self.num_io_partitions = 1
        self.queued_indices = []
        self.loaded_samples = []

        if isinstance(self.dataset, DistConvDataset):
            self.num_io_partitions = self.dataset.num_io_partitions

        self.pool = Pool(processes=num_procs, initializer=DataReader.init_worker, initargs=(self.dataset,))

    @staticmethod
    def init_worker(dataset):
        """
        Initialize worker process.

        Disables the LBANN signal handler since it reports a spurious error
        when the worker process recieves SIGTERM from the master process.
        """
        import signal

        for sig in range(signal.NSIG):
            try:
                signal.signal(sig, signal.SIG_DFL)
                pass
            except:
                pass

        # Process-local storage
        global g_dataset
        g_dataset = dataset


    def terminate(self) -> None:
        """
        Terminate all worker processes.
        """
        self.pool.terminate()

    @staticmethod
    def load_sample(ind) -> Sample:
        """
        Loads the sample from the dataset at the specified index.
        This function must be called from a worker process.

        :param dataset: Dataset
        :type dataset: Dataset
        :param ind: Index to load
        :type ind: int
        :return: Sample
        :rtype: Sample
        """
        return g_dataset[ind]

    def load_next_sample_async(self):
        """
        Submit the next sample index to be loaded to the worker pool.
        """
        self.loaded_samples.append(
            self.pool.apply_async(
                DataReader.load_sample, (self.queued_indices.pop(0),)
            )
        )

    def queue_epoch(self, inds: List[int]) -> None:
        """
        Set the indices to be loaded this epoch and start submitting jobs
        to the worker pool.

        :param inds: List of sample indices
        :type inds: List[int]
        """
        self.queued_indices += inds
        while (
            len(self.loaded_samples) < self.num_procs * self.prefetch_factor
            and len(self.queued_indices) > 0
        ):
            self.load_next_sample_async()

    def get_batch(self, batch_size: int) -> Dict[str, Union[np.ndarray, int]]:
        """
        Return a batch of samples.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :return: Batch of samples and pointers for each input field
        :rtype: Dict[str, Union[np.ndarray, int]]
        """
        samples = []
        for _ in range(batch_size):
            samples.append(self.loaded_samples.pop(0).get())
            if len(self.queued_indices) > 0:
                self.load_next_sample_async()

        batch = {}

        # Note: we return the arrays with the pointers so that they aren't
        # deallocated by the garbage collector.
        batch["sample"] = np.ascontiguousarray(
            [s.sample for s in samples], dtype=self.dtype
        )
        batch["sample_ptr"] = batch["sample"].ctypes.data
        assert (
            batch["sample"].size
            == np.prod(self.sample_dims.sample) * batch_size / self.num_io_partitions
        )

        if hasattr(self.sample_dims, "label"):
            batch["label"] = np.ascontiguousarray(
                [s.label for s in samples], dtype=self.dtype
            )
            batch["label_ptr"] = batch["label"].ctypes.data
            assert batch["label"].size == np.prod(self.sample_dims.label) * batch_size

        if hasattr(self.sample_dims, "response"):
            batch["response"] = np.ascontiguousarray(
                [s.response for s in samples], dtype=self.dtype
            )
            batch["response_ptr"] = batch["response"].ctypes.data
            assert (
                batch["response"].size
                == np.prod(self.sample_dims.response) * batch_size
            )

        return batch


def construct_python_dataset_reader(
    dataset: Dataset,
    dataset_path: Optional[str] = None,
    role: Optional[str] = "train",
    shuffle: Optional[bool] = True,
    fraction_of_data_to_use: Optional[float] = 1.0,
    validation_fraction: Optional[float] = 0.0,
    load_module: Optional[bool] = True,
    prefetch_factor: Optional[int] = 1,
) -> lbann.reader_pb2.Reader:
    """
    Helper function to take a Dataset object, pickle it, save it, and return
    a LBANN data reader protobuf message.

    :param dataset: Dataset
    :type dataset: Dataset
    :param dataset_path: Path to save pickled dataset, defaults to the current working directory
    :type dataset_path: Optional[str], optional
    :param role: LBANN data reader role, defaults to "train"
    :type role: Optional[str], optional
    :param shuffle: Enable shuffle, defaults to True
    :type shuffle: Optional[bool], optional
    :param fraction_of_data_to_use: Fraction of samples to use, defaults to 1.0
    :type fraction_of_data_to_use: Optional[float], optional
    :param validation_fraction: Fraction of samples to use for validation, defaults to 0.0
    :type validation_fraction: Optional[float], optional
    :param load_module: If enabled, adds the dataset class's module to the PYTHONPATH to prevent
        errors when unpickling, defaults to True
    :type load_module: Optional[bool], optional
    :param prefetch_factor: Number of samples to prefetch per data reader process, defaults to 1
    :type prefetch_factor: Optional[int], optional
    :return: LBANN Reader protobuf message
    :rtype: lbann.reader_pb2.Reader
    """
    if dataset_path is None:
        dataset_path = os.path.join(os.getcwd(), f"{role}_dataset.pkl")

    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

    module_dir = None
    if load_module:
        try:
            module_dir = os.path.dirname(inspect.getsourcefile(type(dataset)))
        except:
            pass

    reader = lbann.reader_pb2.Reader(
        name="python_dataset",
        role=role,
        shuffle=shuffle,
        fraction_of_data_to_use=fraction_of_data_to_use,
        validation_fraction=validation_fraction,
        python_dataset=lbann.reader_pb2.PythonDatasetReader(
            dataset_path=dataset_path,
            module_dir=module_dir,
            prefetch_factor=prefetch_factor,
        ),
    )

    return reader
