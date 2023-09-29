from os.path import abspath, dirname, join
import google.protobuf.text_format as txtf
import models.probiesNet as model
import argparse
import lbann
import lbann.contrib.args
import lbann.contrib.launcher

# ==============================================
# Setup and launch experiment
# ==============================================



# Command-line arguments
desc = ('Construct and run ProbiesNet on HRRL PROBIES data. ')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser, 'probiesNet')
parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help='mini-batch size (default: 32)', metavar='NUM')
parser.add_argument(
    '--reader-prototext', action='store', default='probies_v2.prototext', type=str,
    help='data to use (default: probies_v2.prototext, 20K data)')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
#Add reader prototext

lbann.contrib.args.add_optimizer_arguments(parser)
args = parser.parse_args()


# Default data reader
cur_dir = dirname(abspath(__file__))
data_reader_prototext = join(cur_dir,
                             'data',
                             args.reader_prototext)

print("DATA READER ", data_reader_prototext)

images = lbann.Input(data_field='samples')
responses = lbann.Input(data_field='responses')

num_labels = 5

images = lbann.Reshape(images, dims=[1, 300, 300])


pred = model.PROBIESNet(num_labels)(images)

mse = lbann.MeanSquaredError([responses, pred])

# Pearson Correlation
# rho(x,y) = covariance(x,y) / sqrt( variance(x) * variance(y) )
pearson_r_cov = lbann.Covariance([pred, responses],
				   name="pearson_r_cov")

pearson_r_var1 = lbann.Variance(responses,
				 name="pearson_r_var1")

pearson_r_var2 = lbann.Variance(pred,
				name="pearson_r_var2")


pearson_r_mult = lbann.Multiply([pearson_r_var1, pearson_r_var2],
				    name="pearson_r_mult")

pearson_r_sqrt = lbann.Sqrt(pearson_r_mult,
		            name="pearson_r_sqrt")

eps = lbann.Constant(value=1e-07,hint_layer=pearson_r_sqrt)
pearson_r = lbann.Divide([pearson_r_cov, lbann.Add(pearson_r_sqrt,eps)],
			     name="pearson_r")


metrics = [lbann.Metric(mse, name='mse')]
metrics.append(lbann.Metric(pearson_r, name='pearson_r'))

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer()]


layers = list(lbann.traverse_layer_graph([images, responses]))
model = lbann.Model(args.num_epochs,
                    layers=layers,
                    metrics=metrics,
                    objective_function=mse,
                    callbacks=callbacks)



# Load data reader from prototext
data_reader_proto = lbann.lbann_pb2.LbannPB()
with open(data_reader_prototext, 'r') as f:
    txtf.Merge(f.read(), data_reader_proto)
data_reader_proto = data_reader_proto.data_reader

# Setup trainer
trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size)

# Setup optimizer
opt = lbann.Adam(learn_rate=0.0002,beta1=0.9,beta2=0.99,eps=1e-8)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader_proto, opt,
                           lbann_args=" --use_data_store --preload_data_store",
                           job_name=args.job_name,
                           **kwargs)
