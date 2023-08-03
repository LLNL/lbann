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
    data_reader.python.module = 'dataset_CGAN'
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = 'get_sample'
    data_reader.python.num_samples_function = 'num_samples'
    data_reader.python.sample_dims_function = 'sample_dims'

    # Test set data reader
    val_data_reader = message.reader.add()
    val_data_reader.name = 'python'
    val_data_reader.role = 'test'
    val_data_reader.shuffle = False
    val_data_reader.fraction_of_data_to_use = 1.0
    val_data_reader.python.module = 'dataset_CGAN' #use for inference/interpolation
    val_data_reader.python.module_dir = module_dir
    val_data_reader.python.sample_function = 'get_test_sample'
    val_data_reader.python.num_samples_function = 'num_test_samples'
    val_data_reader.python.sample_dims_function = 'sample_dims'
    return message
 

