import combo
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf

# ==============================================
# Setup and launch experiment
# ==============================================

# Default data reader
cur_dir = dirname(abspath(__file__))
data_reader_prototext = join(cur_dir,
                             'data',
                             'combo.prototext')

print("DATA READER ", data_reader_prototext)


def construct_model():
    """Construct LBANN model.

    Pilot1 Combo model

    """
    import lbann

    # Layer graph
    input_ = lbann.Input(target_mode='regression')
    data = lbann.Identity(input_)
    responses = lbann.Identity(input_)

    pred = combo.Combo()(data)
    mse = lbann.MeanSquaredError([responses, pred])

    metrics = [lbann.Metric(mse, name='mse')]
    #lbann.Metric(r2, name='r2')]

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]

    # Construct model
    num_epochs = 100
    layers = list(lbann.traverse_layer_graph(input_))
    return lbann.Model(num_epochs,
                       layers=layers,
                       metrics=metrics,
                       objective_function=mse,
                       callbacks=callbacks)


if __name__ == '__main__':
    import lbann

    mini_batch_size = 256
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size)
    model = construct_model()
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0001,beta1=0.9,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext, 'r') as f:
      txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    status = lbann.run(trainer,model, data_reader_proto, opt,
                       scheduler='lsf',
                       nodes=1,
                       procs_per_node=4,
                       time_limit=360,
                       #setup_only=True,
                       job_name='p1_combo')
    print(status)
