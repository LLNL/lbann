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
    data = lbann.Input(data_field='samples')
    responses = lbann.Input(data_field='responses')

    pred = combo.Combo()(data)
    mse = lbann.MeanSquaredError([responses, pred])

    SS_res = lbann.Reduction(lbann.Square(lbann.Subtract(responses, pred)), mode='sum')

    #SS_tot = var(x) = mean((x-mean(x))^2)
    mini_batch_size = lbann.MiniBatchSize()
    mean = lbann.Divide(lbann.BatchwiseReduceSum(responses), mini_batch_size)
    SS_tot = lbann.Divide(lbann.BatchwiseReduceSum(lbann.Square(lbann.Subtract(responses, mean))), mini_batch_size)
    eps = lbann.Constant(value=1e-07,hint_layer=SS_tot)
    r2 = lbann.Subtract(lbann.Constant(value=1, num_neurons='1'), lbann.Divide(SS_res, lbann.Add(SS_tot,eps)))

    metrics = [lbann.Metric(mse, name='mse')]
    metrics.append(lbann.Metric(r2, name='r2'))

    callbacks = [lbann.CallbackPrint(),
                 lbann.CallbackTimer()]

    # Construct model
    num_epochs = 100
    layers = list(lbann.traverse_layer_graph([data, responses]))
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
                       nodes=2,
                       procs_per_node=4,
                       time_limit=360,
                       #setup_only=True,
                       job_name='p1_combo')
    print(status)
