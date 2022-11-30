import lbann
from gan_model import build_model
from mnist_dataset import make_data_reader

mini_batch_size = 128
trainer = lbann.Trainer(mini_batch_size)

num_epochs = 100
model = build_model(num_epochs)

data_reader = make_data_reader()

opt = lbann.Adam(learn_rate=1e-4, beta1=0., beta2=0.99, eps=1e-8)

lbann.run(trainer, model, data_reader, opt)