import sys
import numpy as np
from utils import str_list


def construct_lc_launcher_args():
    import multiprocessing as mp
    import argparse

    # defaults correspond to the settings needed for training on the moses dataset
    parser = argparse.ArgumentParser(prog="lbann charVAE training")
    parser.add_argument("--account", default="hpcdl")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--scheduler", default="slurm")
    parser.add_argument(
        "--data-module-file",
        help="specifies the module that contains the logic for loading data",
    )
    parser.add_argument(
        "--data-config",
        help="path to a data config file that is used for the construction of the data reader",
    )  # TODO(derek): just put all data relevant info into the config, condense condense condense
    parser.add_argument(
        "--time-limit",
        type=int,
        default=720,
        help="specified time limit in number of minutes",
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--job-name", default="atom_char_rnn_250k")
    parser.add_argument("--embedding-dim", type=int, default=30)
    parser.add_argument("--num-embeddings", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--data-path")
    parser.add_argument("--data-reader-prototext", default=None)
    parser.add_argument("--pad-index", type=int, default=28)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--chunked-data", action="store_true")

    # these are specific to the Trainer object
    parser.add_argument(
        "--procs-per-trainer",
        type=int,
        default=mp.cpu_count(),
        help="number of processes to use per trainer",
    )
    parser.add_argument(
        "--num-parallel-readers",
        type=int,
        default=0,
        help="number of data reader instances",
    )


    # these are the bits and pieces required for loading the model in the moses library...may be useful for evaluation tasks/continuing training/etc
    parser.add_argument("--gamma", type=float, default=0.5, help="")
    parser.add_argument(
        "--hidden", type=int, default=768, help="size of the hidden layer"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="optimizer learning rate to use for training",
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="number of LSTM layers"
    )
    parser.add_argument(
        "--step-size", type=int, default=10, help="learning rate decay step size"
    )
=======
pad_indx = 28
# Sample access functions
def get_sample(index):
    sample = samples[index]
    if len(sample) < dims:
        sample = np.concatenate((sample, np.full(dims-len(sample), pad_indx)))
    else:
        sample = np.resize(sample, dims)
    return sample
>>>>>>> d67f4fbea212560f976e096e1757207ccbe7873a

    # this is just for compatiblity with the moses code
    parser.add_argument("--dropout", type=float, default=0.5, help="")
    return parser.parse_args()


# ==============================================
# Setup and launch experiment
# ==============================================


def construct_model(run_args):
    """Construct LBANN model.

    Initial model for ATOM molecular SMILES generation
    Network architecture and training hyperparameters from
    https://github.com/samadejacobs/moses/tree/master/moses/char_rnn

    """
    import lbann
    import lbann.modules

    sequence_length = run_args.sequence_length
    assert sequence_length is not None

    print("sequence length is {}".format(sequence_length))
    data_layout = "data_parallel"

    # Layer graph
    input = lbann.Input(name="inp_tensor", target_mode="N/A")
    # input = lbann.Input(name='inp_tensor')  # this came from the lbann implementation, above line is a patch
    print(sequence_length)
    x_slice = lbann.Slice(
        input,
        axis=0,
        slice_points=str_list(range(sequence_length + 1)),
        device="CPU",
        name="inp_slice",
    )

    # embedding layer
    emb = []
    embedding_dim = run_args.embedding_dim
    num_embeddings = run_args.num_embeddings

    emb_weights = lbann.Weights(
        initializer=lbann.NormalInitializer(mean=0, standard_deviation=1),
        name="emb_matrix",
    )

    lstm1 = lbann.modules.GRU(size=run_args.hidden, data_layout=data_layout)
    fc = lbann.modules.FullyConnectedModule(
        size=num_embeddings, data_layout=data_layout
    )

    last_output = lbann.Constant(
        value=0.0,
        num_neurons="{}".format(run_args.hidden),
        data_layout=data_layout,
        name="lstm_init_output",
    )

    lstm1_prev_state = [last_output]

    loss = []
    idl = []
    for i in range(sequence_length):
        idl.append(lbann.Identity(x_slice, name="slice_idl_" + str(i), device="CPU"))

    for i in range(sequence_length - 1):

        emb_l = lbann.Embedding(
            idl[i],
            name="emb_" + str(i),
            device="CPU",
            weights=emb_weights,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
        )

        x, lstm1_prev_state = lstm1(emb_l, lstm1_prev_state)
        fc_l = fc(x)
        y_soft = lbann.Softmax(fc_l, name="soft_" + str(i))
        gt = lbann.OneHot(idl[i + 1], size=num_embeddings)
        ce = lbann.CrossEntropy([y_soft, gt], name="loss_" + str(i))
        # mask padding in input
        pad_mask = lbann.NotEqual(
            [idl[i], lbann.Constant(value=run_args.pad_index, num_neurons="1")],
            device="CPU",
        )  # TODO: am adding a .config that will be used to read the value of pad index
        ce_mask = lbann.Multiply([pad_mask, ce], name="loss_mask_" + str(i))
        loss.append(lbann.LayerTerm(ce_mask, scale=1 / (sequence_length - 1)))

    layers = list(lbann.traverse_layer_graph(input))
    # Setup objective function
    weights = set()
    for l in layers:
        weights.update(l.weights)
    obj = lbann.ObjectiveFunction(loss)

    callbacks = [
        lbann.CallbackPrint(),
        lbann.CallbackTimer(),
        lbann.CallbackStepLearningRate(step=run_args.step_size, amt=run_args.gamma),
        lbann.CallbackDumpWeights(basename="weights", epoch_interval=1),
    ]

    # Construct model
    batch_size = run_args.batch_size
    num_epochs = run_args.num_epochs

    return lbann.Model(
        batch_size,
        num_epochs,
        weights=weights,
        layers=layers,
        objective_function=obj,
        callbacks=callbacks,
    )


def construct_data_reader(run_args):
    """
    Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.

    """

    import os.path
    import lbann

    # module_file = os.path.abspath(__file__)
    module_file = os.path.abspath(run_args.data_module_file)
    os.environ["DATA_CONFIG"] = os.path.abspath(run_args.data_config)

    # TODO: here is where I can on the fly switch datasets..probelm is that each one needs their own module but whatever..
    module_name = os.path.splitext(os.path.basename(module_file))[0]
    module_dir = os.path.dirname(module_file)

    print("module_name: {}\tmodule_dir: {}".format(module_name, module_dir))

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    data_reader = message.reader.add()
    data_reader.name = "python"
    data_reader.role = "train"
    data_reader.shuffle = True
    data_reader.percent_of_data_to_use = 1.0
    data_reader.python.module = module_name
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = "get_sample"
    data_reader.python.num_samples_function = "num_samples"
    data_reader.python.sample_dims_function = "sample_dims"

    # in the event we are feeding slices to the script
    if run_args.data_file_pattern is not None:
        data_reader.python.data_file_pattern = run_args.data_file_pattern

    return message


if __name__ == "__main__":

    run_args = construct_lc_launcher_args()
    print(run_args)
    import os
    import lbann

    import lbann.contrib.lc.launcher

    trainer = lbann.Trainer(
        name=None,
        procs_per_trainer=run_args.procs_per_trainer,
        num_parallel_readers=run_args.num_parallel_readers,
    )
    model = construct_model(run_args)
    opt = lbann.Adam(learn_rate=run_args.lr, beta1=0.9, beta2=0.99, eps=1e-8)

    if run_args.chunked_data:

        import os.path
        import lbann
        import google.protobuf.text_format as txtf

        data_reader_proto = lbann.lbann_pb2.LbannPB()
        with open(run_args.data_reader_prototext, "r") as f:
            txtf.Merge(f.read(), data_reader_proto)
        data_reader = data_reader_proto.data_reader

    else:
        data_reader = construct_data_reader(run_args)

    import datetime

    # shamelessly lifted from lbann/blob/develop/python/lbann/launcher/__init__.py
    if "LBANN_EXPERIMENT_DIR" in os.environ:
        work_dir = os.environ["LBANN_EXPERIMENT_DIR"]
    else:
        work_dir = os.path.join(os.getcwd())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(
        work_dir, "{}_{}".format(timestamp, run_args.job_name)
    )
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # dump the config to the experiment_dir so that it can be used to load the model in pytorch (moses codebase)
    import torch

    torch.save(run_args, "{}/{}_config.pt".format(experiment_dir, run_args.job_name))

    status = lbann.contrib.lc.launcher.run(
        trainer,
        model,
        data_reader,
        opt,
        account=run_args.account,
        partition=run_args.partition,
        scheduler=run_args.scheduler,
        time_limit=run_args.time_limit,
        nodes=run_args.nodes,
        job_name=run_args.job_name,
        experiment_dir=experiment_dir,
    )

    print(status)
