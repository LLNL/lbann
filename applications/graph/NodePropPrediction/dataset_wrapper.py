import lbann
import os.path as osp


current_dir = osp.dirname(osp.realpath(__file__))

DATASET_CONFIG = {
    "ARXIV": {
        "num_nodes": 169343,
        "num_edges": 1166243,
        "input_features": 128,
    }
}


def make_data_reader(dataset):
    reader = lbann.reader_pb2.DataReader()
    reader.name = "python"
    reader.role = "train"
    reader.shuffle = True
    reader.percent_of_data_to_use = 1.0
    reader.python.module = f"{dataset}_dataset"
    reader.python.module_dir = osp.join(current_dir, "datasets")
    reader.python.sample_function = "get_train_sample"
    reader.python.num_samples_function = "num_train_samples"
    reader.python.sample_dims_function = "sample_dims"
    return reader
