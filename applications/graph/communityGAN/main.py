import lbann
import lbann.contrib.launcher

import model
import data

# Options
motif_size = 4
walk_length = 20
num_vertices = 1234
embed_dim = 128
learn_rate = 1e-2
mini_batch_size = 512
num_epochs = 100

# Construct LBANN objects
trainer = lbann.Trainer(
    mini_batch_size=mini_batch_size,
    num_parallel_readers=0,
)
model_ = model.make_model(
    motif_size,
    walk_length,
    num_vertices,
    embed_dim,
    learn_rate,
    num_epochs,
)
optimizer = lbann.SGD(learn_rate=learn_rate)
data_reader = data.make_data_reader()

# Run LBANN
lbann.contrib.launcher.run(
    trainer,
    model_,
    data_reader,
    optimizer,
    job_name='lbann_communitygan',
)
