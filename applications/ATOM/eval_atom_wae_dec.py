import argparse
import datetime
import os
#import os.path
from os.path import abspath, dirname, join
import sys

from google.protobuf import text_format as txtf
import json
import numpy as np
import models.wae_stack as molwae

import lbann
import lbann.contrib.launcher
import lbann.modules
from lbann.util import str_list

def list2str(l):
    return ' '.join(l)

lbann_exe = abspath(lbann.lbann_exe())
lbann_exe = join(dirname(lbann_exe), 'lbann_inf')

def construct_lc_launcher_args():

    # defaults correspond to the settings needed for training on the moses dataset
    parser = argparse.ArgumentParser(prog="lbann ATOM WAE training")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default="hpcdl")
    parser.add_argument("--scheduler", type=str, default="slurm")
    parser.add_argument(
        "--data-module-file",
        default="dataset.py",
        help="specifies the module that contains the logic for loading data",
    )
    parser.add_argument(
        "--data-config",
        default=os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "mpro_data_config.json"
        ),
        help="path to a data config file that is used for the construction of python data reader",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=720,
        help="specified time limit in number of minutes",
    )
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--job-name", default="atom_wae")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-embeddings", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--data-reader-prototext", default=None)
    parser.add_argument("--pad-index", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--dump-model-dir", type=str, default=None)
    parser.add_argument("--dump-outputs-dir", type=str, default="output")
    parser.add_argument("--dump-outputs-interval", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-io-threads", type=int, default=11)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--delimiter", default="c")
    parser.add_argument("--no-header", type=bool, default=True)

    # these are specific to the Trainer object
    parser.add_argument(
        "--procs-per-trainer",
        type=int,
        default=0,
        help="number of processes to use per trainer",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="optimizer learning rate to use for training",
    )
    return parser.parse_args()

# ==============================================
# Setup and launch experiment
# ==============================================

def construct_model(run_args):
    """Construct LBANN model.

    Initial model for ATOM molecular VAE

    """
    import lbann

    pad_index = run_args.pad_index
    assert pad_index is not None

    #sequence_length = run_args.sequence_length
    sequence_length = 102
    assert sequence_length is not None

    print("sequence length is {}".format(sequence_length))
    data_layout = "data_parallel"
    # Layer graph
    input_ = lbann.Input(target_mode='N/A',name='inp_data')
    inp_slice = lbann.Slice(input_, axis=0, slice_points="0 102 230",name='inp_slice')
    inp_smile = lbann.Identity(inp_slice,name='inp_smile')
    z = lbann.Identity(inp_slice, name='z') #param not used
    #input_ = lbann.Identity(lbann.Input(name='inp',target_mode="N/A"), name='inp1')
    vae_loss= []
    input_feature_dims = sequence_length

    embedding_size = run_args.embedding_dim
    dictionary_size = run_args.num_embeddings
    assert embedding_size is not None
    assert dictionary_size is not None

    save_output = True if run_args.dump_outputs_dir else False

    #print("Inp smile len ", len(inp_smile), "z len ",  len(z))
    print("save output? ", save_output, "out dir ",  run_args.dump_outputs_dir)
    #z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims="128")
    x = lbann.Slice(inp_smile, slice_points=str_list([0, input_feature_dims]))
    x = lbann.Identity(x)
    waemodel = molwae.MolWAE(input_feature_dims,
                           dictionary_size,
                           embedding_size,
                           pad_index,save_output)
    x_emb = lbann.Embedding(
            x,
            num_embeddings=waemodel.dictionary_size,
            embedding_dim=waemodel.embedding_size,
            name='emb',
            weights=waemodel.emb_weights
    )

    
    pred, arg_max = waemodel.forward_decoder(x_emb,z)

    recon = waemodel.compute_loss(x, pred)



    vae_loss.append(recon)

    layers = list(lbann.traverse_layer_graph(input_))
    # Setup objective function
    weights = set()
    for l in layers:
      weights.update(l.weights)
    #l2_reg = lbann.L2WeightRegularization(weights=weights, scale=1e-4)

    #vae_loss.append(l2_reg)
    print("LEN vae loss ", len(vae_loss))

    obj = lbann.ObjectiveFunction(vae_loss)

    # Initialize check metric callback
    metrics = [lbann.Metric(recon, name='recon')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]


    #Dump output (activation) for post processing
    pred_tensor = lbann.Concatenation(arg_max, name='pred_tensor')
    conc_out = lbann.Concatenation([input_,pred_tensor], name='conc_out')
    callbacks.append(lbann.CallbackDumpOutputs(batch_interval=run_args.dump_outputs_interval, 
                       execution_modes='test', 
                       directory=run_args.dump_outputs_dir,
                       layers=f'{conc_out.name}'))
    # Construct model
    return lbann.Model(run_args.num_epochs,
                       weights=weights,
                       layers=layers,
                       objective_function=obj,
                       metrics=metrics,
                       callbacks=callbacks)


def construct_data_reader(run_args):
    """
    Construct Protobuf message for Python data reader.

    The Python data reader will import this Python file to access the
    sample access functions.

    """

    module_file = os.path.abspath(run_args.data_module_file)
    os.environ["DATA_CONFIG"] = os.path.abspath(run_args.data_config)

    module_name = os.path.splitext(os.path.basename(module_file))[0]
    module_dir = os.path.dirname(module_file)

    print("module_name: {}\tmodule_dir: {}".format(module_name, module_dir))

    # Base data reader message
    message = lbann.reader_pb2.DataReader()

    # Training set data reader
    data_reader = message.reader.add()
    data_reader.name = "python"
    data_reader.role = "test"
    data_reader.shuffle = False
    data_reader.percent_of_data_to_use = 1.0
    #data_reader.validation_percent = 0.1
    data_reader.python.module = module_name
    data_reader.python.module_dir = module_dir
    data_reader.python.sample_function = "get_sample"
    data_reader.python.num_samples_function = "num_samples"
    data_reader.python.sample_dims_function = "sample_dims"

    return message


def main():
    run_args = construct_lc_launcher_args()

    # add data_config data
    # and do not overwrite args if data_reader_prototext is enabled
    if os.path.isfile(run_args.data_config) and not run_args.data_reader_prototext:
        with open(run_args.data_config, "r") as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(run_args, k, v)

    trainer = lbann.Trainer(
        run_args.batch_size,
        #name=None,
        #procs_per_trainer=run_args.procs_per_trainer,
    )

    # define data_reader
    if run_args.data_reader_prototext:
        print("Using data_reader_prototext")
        assert run_args.sequence_length is not None
        assert run_args.vocab is not None

        data_reader_proto = lbann.lbann_pb2.LbannPB()
        with open(run_args.data_reader_prototext, "r") as f:
            txtf.Merge(f.read(), data_reader_proto)
        data_reader = data_reader_proto.data_reader
    else:
        data_reader = construct_data_reader(run_args)

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

    # model and optimizer
    model = construct_model(run_args)
    opt = lbann.Adam(learn_rate=run_args.lr, beta1=0.9, beta2=0.99, eps=1e-8)

    # dump the config to the experiment_dir so that it can be used to load the model in pytorch (moses codebase)
    ppn = 4 if run_args.scheduler == "lsf" else 2
    print("args:\n" + str(run_args))
    if(run_args.scheduler == 'slurm'):
      import torch
      torch.save(run_args, "{}/{}_config.pt".format(experiment_dir, run_args.job_name))

    m_lbann_args=f"--load_model_weights_dir_is_complete --load_model_weights_dir={run_args.dump_model_dir} --vocab={run_args.vocab} --num_samples={run_args.num_samples} --sequence_length={run_args.sequence_length}  --num_io_threads={run_args.num_io_threads} --no_header={run_args.no_header} --delimiter={run_args.delimiter}"
    if(run_args.data_reader_prototext):
      m_lbann_args = " ".join((m_lbann_args, " --use_data_store --preload_data_store "))

    status = lbann.contrib.launcher.run(
        trainer,
        model,
        data_reader,
        opt,
        lbann_exe,
        partition=run_args.partition,
        scheduler=run_args.scheduler,
        account=run_args.account,
        time_limit=run_args.time_limit,
        nodes=run_args.nodes,
        procs_per_node=ppn,
        #batch_job = True,
        #setup_only = True,
        job_name=run_args.job_name,
        experiment_dir=experiment_dir,
        lbann_args = m_lbann_args,
        #turn on for tensor core
        environment = {
            'LBANN_USE_CUBLAS_TENSOR_OPS' : 1,
            'LBANN_USE_CUDNN_TENSOR_OPS' : 1,
        },
    )

    print("LBANN launcher status:\n" + str(status))


if __name__ == "__main__":
    sys.exit(main())
