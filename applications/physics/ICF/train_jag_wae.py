import jag_models
from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf

# ==============================================
# Setup and launch experiment
# ==============================================

# Default data reader
model_zoo_dir = dirname(dirname(abspath(__file__)))
data_reader_prototext = join(model_zoo_dir,
                             'data',
                             'jag_100Kdata.prototext')

if __name__ == '__main__':
    import lbann

    y_dim = 16399 #image+scalar shape
    z_dim = 20  #Latent space dim
    num_epochs = 100
    mini_batch_size = 128
    trainer = lbann.Trainer(mini_batch_size=mini_batch_size,
                            serialize_io=True)
    model = jag_models.construct_jag_wae_model(y_dim=y_dim,
                                               z_dim=z_dim,
                                               num_epochs=num_epochs)
    # Setup optimizer
    opt = lbann.Adam(learn_rate=0.0001,beta1=0.9,beta2=0.99,eps=1e-8)
    # Load data reader from prototext
    data_reader_proto = lbann.lbann_pb2.LbannPB()
    with open(data_reader_prototext, 'r') as f:
      txtf.Merge(f.read(), data_reader_proto)
    data_reader_proto = data_reader_proto.data_reader

    status = lbann.run(trainer,model, data_reader_proto, opt,
                       scheduler='slurm',
                       nodes=1,
                       procs_per_node=1,
                       time_limit=360,
                       job_name='jag_wae')
    print(status)
