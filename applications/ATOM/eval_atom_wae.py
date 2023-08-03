import argparse
import datetime
import os
#import os.path
from os.path import abspath, dirname, join
import sys

from google.protobuf import text_format as txtf
import json
import numpy as np
import models.wae as molwae

import lbann
import lbann.contrib.launcher
import lbann.modules

def list2str(l):
    return ' '.join(l)

lbann_exe = abspath(lbann.lbann_exe())
lbann_exe = join(dirname(lbann_exe), 'lbann_inf')

def construct_lc_launcher_args():

    # defaults correspond to the settings needed for training on the moses dataset
    parser = argparse.ArgumentParser(prog="lbann ATOM VAE training")
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
            os.path.abspath(os.path.dirname(__file__)), "zinc_data_config.json"
        ),
        help="path to a data config file that is used for the construction of python data reader",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=720,
        help="specified time limit in number of minutes",
    )
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--job-name", default="atom_wae")
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--num-embeddings", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--z-dim", type=int, default=512, help="latent space dim")
    parser.add_argument("--lamda", type=float, default=0.001, help="weighting of adversarial loss")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--data-reader-prototext", default=None)
    #parser.add_argument("--data-filedir", default=None)
    #parser.add_argument("--data-filename", default=None)
    parser.add_argument("--pad-index", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--dump-model-dir", type=str, default=None)
    parser.add_argument("--dump-outputs-dir", type=str, default="output")
    parser.add_argument("--dump-outputs-interval", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-io-threads", type=int, default=11)
    parser.add_argument("--vocab", default=None)

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

    print("Dump model dir ", run_args.dump_model_dir)
    assert run_args.dump_model_dir, "evaluate script asssumes a pretrained WAE model"
    pad_index = run_args.pad_index
    assert pad_index is not None

    sequence_length = run_args.sequence_length
    assert sequence_length is not None

    print("sequence length is {}".format(sequence_length))
    data_layout = "data_parallel"
    # Layer graph
    input_ = lbann.Identity(lbann.Input(name='inp',data_field='samples'), name='inp1')
    wae_loss= []
    input_feature_dims = sequence_length

    embedding_size = run_args.embedding_dim
    dictionary_size = run_args.num_embeddings
    assert embedding_size is not None
    assert dictionary_size is not None

    save_output = True if run_args.dump_outputs_dir else False

    print("save output? ", save_output, "out dir ",  run_args.dump_outputs_dir)
    z = lbann.Gaussian(mean=0.0,stdev=1.0, neuron_dims=run_args.z_dim)

    waemodel = molwae.MolWAE(input_feature_dims,
                           dictionary_size,
                           embedding_size,
                           pad_index,run_args.z_dim,save_output)
    recon, d1_real, d1_fake, d_adv, arg_max = waemodel(input_,z)


    zero  = lbann.Constant(value=0.0,num_neurons=[1],name='zero')
    one  = lbann.Constant(value=1.0,num_neurons=[1],name='one')

    d1_real_bce = lbann.SigmoidBinaryCrossEntropy([d1_real,one],name='d1_real_bce')
    d1_fake_bce = lbann.SigmoidBinaryCrossEntropy([d1_fake,zero],name='d1_fake_bce')
    d_adv_bce = lbann.SigmoidBinaryCrossEntropy([d_adv,one],name='d_adv_bce')

    wae_loss.append(recon)

    layers = list(lbann.traverse_layer_graph(input_))
    # Setup objective function
    weights = set()
    src_layers = []
    dst_layers = []
    for l in layers:
      if(l.weights and "disc0" in l.name and "instance1" in l.name):
        src_layers.append(l.name)
      #freeze weights in disc2
      if(l.weights and "disc1" in l.name):
        dst_layers.append(l.name)
        for idx in range(len(l.weights)):
          l.weights[idx].optimizer = lbann.NoOptimizer()
      weights.update(l.weights)
    l2_weights = [w for w in weights if not isinstance(w.optimizer, lbann.NoOptimizer)]
    l2_reg = lbann.L2WeightRegularization(weights=l2_weights, scale=1e-4)

    wae_loss.append(d1_real_bce)
    wae_loss.append(d_adv_bce)
    wae_loss.append(d1_fake_bce)
    wae_loss.append(l2_reg)
    print("LEN wae loss ", len(wae_loss))

    obj = lbann.ObjectiveFunction(wae_loss)

    # Initialize check metric callback
    metrics = [lbann.Metric(d_adv_bce, name='adv_loss'),
               lbann.Metric(recon, name='recon')
                ]

    callbacks = [lbann.CallbackPrint(),
                 #lbann.CallbackStepLearningRate(step=10, amt=0.5),
                 lbann.CallbackTimer()]


    callbacks.append(lbann.CallbackReplaceWeights(source_layers=list2str(src_layers),
                                 destination_layers=list2str(dst_layers),
                                 batch_interval=2))

    #Dump output (activation) for post processing
    if(run_args.dump_outputs_dir):
      pred_tensor = lbann.Concatenation(arg_max, name='pred_tensor')
      callbacks.append(lbann.CallbackDumpOutputs(batch_interval=run_args.dump_outputs_interval,
                       execution_modes='test',
                       directory=run_args.dump_outputs_dir,
                       layers=f'inp pred_tensor {waemodel.q_mu.name}'))
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
    data_reader.role = "train"
    data_reader.shuffle = True
    data_reader.fraction_of_data_to_use = 1.0
    data_reader.validation_fraction = 0.1
    data_reader.tournament_fraction = 0.1
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

    m_lbann_args=f"--load_model_weights_dir_is_complete --load_model_weights_dir={run_args.dump_model_dir} --vocab={run_args.vocab} --num_samples={run_args.num_samples} --sequence_length={run_args.sequence_length}  --num_io_threads={run_args.num_io_threads}"
    if(run_args.data_reader_prototext):
      m_lbann_args = " ".join((m_lbann_args, " --use_data_store --preload_data_store "))
    if(run_args.procs_per_trainer):
      m_lbann_args = " ".join((m_lbann_args, f"--procs_per_trainer={run_args.procs_per_trainer}"))

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
        batch_job = True,
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
