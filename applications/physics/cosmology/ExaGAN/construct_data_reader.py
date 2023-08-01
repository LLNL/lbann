#Different data reader options for cosmoflow
import dataset3D
import lbann

def construct_python_data_reader():
    """Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.

    """
    import os.path
    module_file = os.path.abspath(__file__)
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    module_dir = os.path.dirname(module_file)

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    data_reader = message.reader.add()
    data_reader.name = 'python'
    data_reader.role = 'train'
    data_reader.shuffle = True
    data_reader.fraction_of_data_to_use = 1.0
    data_reader.validation_fraction = 0.2
    data_reader.python.module = 'dataset3D'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    return message


def create_hdf5_data_reader(
        train_path, val_path, test_path, num_responses=4):
    """Create a data reader for CosmoFlow.

    Args:
        {train, val, test}_path (str): Path to the corresponding dataset.
        num_responses (int): The number of parameters to predict.
    """

    reader_args = [
        {"role": "train", "data_filename": train_path},
        {"role": "validate", "data_filename": val_path},
        {"role": "test", "data_filename": test_path},
    ]

    for reader_arg in reader_args:
        reader_arg["data_file_pattern"] = "{}/*.hdf5".format(
            reader_arg["data_filename"])
        reader_arg["hdf5_key_data"] = "full"
        #reader_arg["hdf5_key_responses"] = "unitPar"
        #reader_arg["num_responses"] = num_responses
        reader_arg.pop("data_filename")

    readers = []
    for reader_arg in reader_args:
        reader = lbann.reader_pb2.Reader(
            name="hdf5",
            shuffle=(reader_arg["role"] != "test"),
            validation_fraction=0,
            absolute_sample_count=0,
            fraction_of_data_to_use=1.0,
            disable_labels=True,
            disable_responses=True,
            scaling_factor_int16=1.0,
            **reader_arg)

        readers.append(reader)

    return lbann.reader_pb2.DataReader(reader=readers)
