import lbann 
import lbann.contrib.launcher 
import lbann.contrib.args

import argparse 
import os 

import Sparse_Graph_Trainer 
import Dense_Graph_Trainer
import data.PROTEINS

desc = (" Training a Graph Convolutional Model using LBANN" )

parser = argparse.ArgumentParser(description=desc)

lbann.contrib.args.add_scheduler_arguments(parser, 'GCN_TEST')
lbann.contrib.args.add_optimizer_arguments(parser) 

parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (deafult: 100)', metavar='NUM')

parser.add_argument(
    '--mini-batch-size', action='store',default=32, type=int,
    help="mini-batch size (default: 32)", metavar='NUM')

parser.add_argument(
    '--model', action = 'store', default='GCN', type=str,
    help="The type of model to use", metavar='NAME')

args = parser.parse_args()


kwargs = lbann.contrib.args.get_scheduler_kwargs(args) 

num_epochs = args.num_epochs
mini_batch_size = args.mini_batch_size 
job_name = args.job_name
model_arch = args.model


## Get Model

data_reader = None
if (model_arch == 'GRAPH'):
    model = Sparse_Graph_Trainer.make_model(kernel_type = 'Graph',
                                            num_epochs = num_epochs)
elif(model_arch=='GIN'):
    model = Sparse_Graph_Trainer.make_model(kernel_type = 'GIN',
                                            num_epochs = num_epochs)
elif(model_arch=='GATEDGRAPH'):
    model = Sparse_Graph_Trainer.make_model(dataset = 'PROTEINS',
                                            kernel_type = 'GatedGraph',
                                            num_epochs = num_epochs)
elif (model_arch =='DGCN'):
    model = Dense_Graph_Trainer.make_model(kernel_type = 'GCN',
                                           num_epochs = num_epochs)
    data_reader = data.PROTEINS.make_data_reader("Dense")
elif (model_arch == 'DGRAPH'):
    model = Dense_Graph_Trainer.make_model(kernel_type = 'Graph',
                                           num_epochs = num_epochs)
    data_reader = data.PROTEINS.make_data_reader("Dense")

else:   
    model = Sparse_Graph_Trainer.make_model(kernel_type = 'GCN',
                                            num_epochs=num_epochs)

if data_reader is None:
    data_reader = data.PROTEINS.make_data_reader()

optimizer = lbann.SGD(learn_rate = 1e-3)

#add logic for choosing a dataset 

trainer = lbann.Trainer(mini_batch_size = mini_batch_size)


lbann.contrib.launcher.run(trainer, model, data_reader, optimizer,
                           job_name = job_name,
                           **kwargs)
