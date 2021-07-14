import argparse
import lbann
import lbann.contrib.launcher
import lbann.contrib.args
from SGCNN.model import SGCNN
from data_util import slice_graph_data
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
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
data_dir = os.path.dirname(os.path.realpath(__file__))


def make_data_reader(module_name='SIM_SGCNN_Dataset',
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
    _reader.shuffle = False  # Turn off shuffle for debugging
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = module_name
    _reader.python.module_dir = module_dir
    _reader.python.sample_function = sample_function
    _reader.python.num_samples_function = num_samples_function
    _reader.python.sample_dims_function = sample_dims

    return reader


def make_model(num_epochs):
    num_nodes = 100
    sgcnn_model = SGCNN(
                  num_nodes=num_nodes)

    input_ = lbann.Input(target_mode='N/A')
    X, cov_adj, noncov_adj, edge_ft, edge_adj, ligand_only, target = \
        slice_graph_data(input_,
                         num_nodes=num_nodes)

    predicted = sgcnn_model(X,
                            cov_adj,
                            noncov_adj,
                            edge_ft,
                            edge_adj,
                            ligand_only)

    loss = lbann.MeanAbsoluteError([predicted, target], name='MAE_loss')
    layers = lbann.traverse_layer_graph(input_)
    metrics = [lbann.Metric(loss, name='MAE')]
    # Prints initial Model after Setup
    print_model = lbann.CallbackPrintModelDescription()
    # Prints training progress
    training_output = lbann.CallbackPrint(interval=1,
                                          print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()
    callbacks = [training_output, gpu_usage, timer]
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
