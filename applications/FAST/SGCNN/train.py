import argparse
import lbann
import lbann.contrib.launcher
import lbann.contrib.args
from lbann.utils import str_list
from model import SGCNN
import os
import os.path as osp

desc = ("Training SGCNN on a Protein-ligand graphs using LBANN")
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--mini-batch-size",
                    action='store', default=32, type=int,
                    help="Mini batch size {default: 32}", metavar='NUM')
parser.add_argument("--num-epochs",
                    action='store', default=1, type=int,
                    help="Number of epochs {default: 1}", metavar='NUM')
parser.add_argument("--job-name",
                    action='store', default="SGCNN", type=str,
                    help="Job name {default: SGCNN}", metavar='NAME')
lbann.contrib.args.add_scheduler_arguments(parser)
args = parser.parse_args()
data_dir = os.path.dirname(os.path.realpath(__file__))


def make_data_reader(module_name='SimDataset',
                     module_dir=None,
                     sample_function='get_train',
                     num_samples_function='num_train_samples',
                     sample_dims='sample_dims'):
    module_dir = (module_dir if module_dir else
                  osp.dirname(osp.realpath(__file__)))
    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = False  # Turn off shuffle for debugging
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = module_name
    _reader.python.module_dir = module_dir
    _reader.python.sample_function = sample_function
    _reader.python.num_samples_function = num_samples_function
    _reader.python.sample_dims_function = sample_dims

    return reader


def slice_data(input_layer,
               num_nodes=50,
               node_features=19,
               edge_features=1):
    # Slice points for node features
    node_fts = [i * node_features for i in range(num_nodes)]
    node_ft_end = node_fts[-1]
    # Slice points for covalent adj matrix
    cov_adj_mat_end = node_ft_end + (num_nodes **2)
    cov_adj_mat = [node_ft_end, cov_adj_mat_end]
    # Slice points for noncovalent adj matrix
    noncov_adj_mat_end = cov_adj_mat_end + (num_nodes **2)
    noncov_adj_mat = [cov_adj_mat_end, noncov_adj_mat_end]
    # Slice points for edge features
    num_edges = int(num_nodes * (num_nodes - 1) / 2)
    edge_fts = [(noncov_adj_mat_end+(i+1)*edge_features) for i in range(num_edges)]
    edge_ft_end = edge_fts[-1]
    # Slice points for edge_adjacencies
    # This should be num_nodes * (num_nodes ** 2)
    prev_end = edge_ft_end
    edge_adj_list = []
    for node in range(num_nodes):
        edge_adj = [(prev_end + (i+1)*num_nodes) for i in range(num_nodes)]
        prev_end = edge_adj[-1]
        edge_adj_list.extend(edge_adj)
    # Slice points for ligand_only mat
    edge_adj_end = edge_adj_list[-1]
    ligand_only_end = edge_adj_end + (num_nodes **2)
    ligand_only = [edge_adj_end, ligand_only_end]
    # Slice for binding energy target
    target_end = ligand_only_end + 1
    target = [ligand_only, target_end]
    slice_points = []
    slice_points.extend(node_fts)
    slice_points.extend(cov_adj_mat)
    slice_points.extend(noncov_adj_mat)
    slice_points.extend(edge_fts)
    slice_points.extend(ligand_only)
    slice_points.extend(target)

    sliced_input = lbann.Slice(input_layer, slice_points=str_list(slice_points))

    node_fts = [lbann.Identity(sliced_input, name="Node_{}".format(i)) for i in range(num_nodes)]
    cov_adj_mat = lbann.Identity(sliced_input, name="Covalent_Adj")
    noncov_adj_mat = lbann.Identity(sliced_input, name="NonCovalent_Adj")
    edge_fts = [lbann.Identity(sliced_input, name="Edge_{}".format(i)) for i in range(num_edges)]
    edge_adj = [lbann.Identity(sliced_input, name="Adj_Mat_{}".format(i)) for i in range(num_nodes)]
    ligand_ID = lbann.Identiy(sliced_input, name="Ligand_only_nodes")
    target = lbann.Identity(sliced_input, name="Target")

    node_fts = [lbann.Reshape(i, dims=str_list([1, node_features])) for i in node_fts]
    cov_adj_mat = lbann.Reshape(cov_adj_mat, dims=str_list([num_nodes, num_nodes]))
    noncov_adj_mat = lbann.Reshape(noncov_adj_mat, dims=str_list([num_nodes, num_nodes]))
    edge_features = \
        [lbann.Reshape(i, dims=str_list([1, edge_features])) for i in edge_fts]
    edge_adj = \
        [lbann.Reshape(i, dims=str_list([num_nodes, num_nodes])) for i in edge_adj]
    ligand_only = lbann.Reshape(ligand_only, dims=str_list([num_nodes, num_nodes]))
    target = lbann.Reshape(target, dims="1")
    return node_fts, cov_adj_mat, noncov_adj_mat, edge_features, edge_adj, ligand_ID, target


def make_model(num_epochs):
    sgcnn_model = SGCNN()

    input_ = lbann.Input(target_mode='N/A')
    X, cov_adj, noncov_adj, edge_ft, edge_adj, ligand_only, target = slice_data(input_)

    predicted = sgcnn_model(X,
                            cov_adj,
                            noncov_adj,
                            edge_ft,
                            edge_adj,
                            ligand_only)

    loss = lbann.MeanSquaredError(predicted, target)
    layers = lbann.traverse_layer_graph(input_)

    # Prints initial Model after Setup
    print_model = lbann.CallbackPrintModelDescription()
    # Prints training progress
    training_output = lbann.CallbackPrint(interval=1,
                                          print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()
    callbacks = [print_model, training_output, gpu_usage, timer]
    model = lbann.Model(num_epochs,
                        layers=layers,
                        objective_function=loss,
                        metrics=metrics,
                        callbacks=callbacks
                        )
    return model


def main():
    optimizer = lbann.Adam(learn_rate=1e-3,
                           beta1=0.9,
                           beta2=0.99,
                           eps=1e-8)
    mini_batch_size = args.mini_batch_size
    num_epochs = args.num_epochs
    data_reader = make_data_reader()
    model = make_model(num_epochs)
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

    lbann.contrib.launcher.run(trainer, model, data_reader, optimizer,
                               job_name=job_name,
                               **kwargs)
if __name__ == '__main__':
    main()
