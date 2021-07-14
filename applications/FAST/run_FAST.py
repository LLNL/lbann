import argparse
import lbann
import os
import os.path as osp
from CNN3D.model import CNN3D
from SGCNN.model import SGCNN
import lbann.contrib.launcher
import lbann.contrib.args
import lbann.modules as nn
from data_util import slice_FAST_data


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
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
data_dir = os.path.dirname(os.path.realpath(__file__))


def make_data_reader(module_name='SIM_FAST_Dataset',
                     module_dir=None,
                     sample_function='get_train',
                     num_samples_function='num_train_samples',
                     sample_dims='sample_dims'):
    data_dir = osp.join(osp.dirname(os.path.realpath(__file__)), 'data')

    module_dir = (module_dir if module_dir else data_dir)

    reader = lbann.reader_pb2.DataReader()
    _reader = reader.reader.add()
    _reader.name = 'python'
    _reader.role = 'train'
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = 'dataset'
    _reader.python.module_dir = module_dir
    _reader.python.sample_function = 'get_train'
    _reader.python.num_samples_function = 'num_train_samples'
    _reader.python.sample_dims_function = 'sample_dims'


def make_model(num_epochs):
    input_ = lbann.Input(target_mode='N/A')

    data = lbann.Identity(input_)

    grid, nodes, cov_adj, noncov_adj, edge_ft, edge_adj, ligand_only, y = \
        slice_FAST_data(data)

    cnn_model = CNN3D()
    gcn_model = SGCNN()

    grid_rep = cnn_model(grid)

    graph_rep = gcn_model(nodes,
                          cov_adj,
                          noncov_adj,
                          edge_ft,
                          edge_adj,
                          ligand_only)

    graph_linear = nn.FullyConnectedModule(5)
    grid_linear = nn.FullyConnectedModule(5)
    graph_linear = graph_linear(graph_rep)
    grid_linear = grid_linear(grid_rep)

    nn_2 = nn.FullyConnectedModule(10)
    nn_out = nn.FullyConnectedModule(1)

    x = lbann.Concatenate([graph_rep, grid_rep, graph_linear, grid_linear])
    x = nn_2(x)
    y_pred = nn_out(x)

    loss = lbann.MeanAbsoluteError([y_pred, y], name='MAE_loss')
    layers = lbann.traverse_layer_graph(input_)
    metrics = [lbann.Metric(loss, name='MAE')]
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
    job_name = args.job_name
    data_reader = make_data_reader()
    model = make_model(num_epochs)
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

    lbann.contrib.launcher.run(trainer, model, data_reader, optimizer,
                               job_name=job_name,
                               **kwargs)


if __name__ == '__main__':
    main()
