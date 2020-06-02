import argparse 
import lbann 
import data.mnist 

# ----------------------------------
# Command-line arguments
# ----------------------------------

## TO DO. Add more CLI arguemnts for Model storage and Generated Data

desc = ("Training 3D-CAE on 4D MOF Data using LBANN")

parser = argparse.ArgumentParser(description = desc)

parser.add_argument(
    '--partition', action='store', type=str,
    help='scheduler partition', metavar='NAME')
parser.add_argument(
    '--account', action='store', type=str,
    help='scheduler account', metavar='NAME')
parser.add_argument(
	'--zdim', action='store',default = 2048, type=int, 
	help="dimensionality of latent space (dedfault: 2048)", metavar = 'NUM')
parser.add_argument(
	'--atoms', action='store', default = 11,type=int, 
	help="Number of atom species (default: 11)", metavar = 'NUM')
parser.add_argument(
    '--job-name', action='store', default='mofae', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')
parser.add_argument(
    '--num-nodes', action='store', default=4, type=int,
    help='number of nodes (default: 4)', metavar='NUM')
parser.add_argument(
    '--ppn', action='store', default=4, type=int,
    help='processes per node (default: 4)', metavar='NUM') ##Change to 1 for local machine / openhpc


args = parser.parse_args()


latent_dim = args.zdim
number_of_atoms = args.atoms


# ----------------------------------
# Construct Graph
# ----------------------------------
input_ = lbann.Input( target_mode = "reconstruction")


tensors = lbann.Identity(input_)

# Input tensor shape is  32x32x32x(number_of_atoms)

# Encoder 
x = lbann.Convolution(tensors,
					  num_dims = 3, 
					  num_out_channels = latent_dim // 8, 
					  num_groups = 1, 
					  conv_dims_i = 4, 
					  conv_strides_i = 2, 
					  conv_dilations_i = 1,
					  has_bias = True)
x = lbann.BatchNormalization(x)
x = lbann.LeakyRelu(x)

# Shape: 16x16x16x(latent_dim // 8)

x = lbann.Convolution(x,
					  num_dims = 3, 
					  num_out_channels = latent_dim // 4, 
					  num_groups = 1, 
					  conv_dims_i = 4, 
					  conv_strides_i = 2, 
					  conv_dilations_i = 1,
					  has_bias = True)
x = lbann.BatchNormalization(x)
x = lbann.LeakyRelu(x)

# Shape: 8x8x8x(latent_dim // 4)

x = lbann.Convolution(x,
					  num_dims = 3, 
					  num_out_channels = latent_dim // 2, 
					  num_groups = 1, 
					  conv_dims_i = 4, 
					  conv_strides_i = 2, 
					  conv_dilations_i = 1,
					  has_bias = True)
x = lbann.BatchNormalization(x)
x = lbann.LeakyRelu(x)

# Shape: 4x4x4x(latent_dim // 2)

x = lbann.Convolution(x,
					  num_dims = 3, 
					  num_out_channels = latent_dim, 
					  num_groups = 1, 
					  conv_dims_i = 4, 
					  conv_strides_i = 2, 
					  conv_dilations_i = 1,
					  has_bias = True)
x = lbann.BatchNormalization(x)
x = lbann.LeakyRelu(x)

# Shape: 2x2x2x(latent_dim)

encoded = lbann.Convolution(tensors,
					  num_dims = 3, 
					  num_out_channels = latent_dim, 
					  num_groups = 1, 
					  conv_dims_i = 2, 
					  conv_strides_i = 2, 
					  conv_dilations_i = 1,
					  has_bias = True,
					  name ="encoded")

# Shape: 1x1x1x(latent_dim)


# Decoder 

x = lbann.Deconvolution(encoded,
						num_dims = 3,
						num_out_channels = number_of_atoms * 16,
						num_groups = 1,
						conv_dims_i = 4,
						conv_pads_i = 0,
						conv_strides_i = 2,
						conv_dilations_i = 1
						has_bias = True
						)
x = lbann.BatchNormalization(x) #Is thia 3D batch norm? 
x = lbann.Tanh(x)
x = lbann.Deconvolution(x,
						num_dims = 3,
						num_out_channels = number_of_atoms * 4,
						num_groups = 1,
						conv_dims_i = 4,
						conv_pads_i = 1,
						conv_strides_i = 2,
						conv_dilations_i = 1
						has_bias = True
						)
x = lbann.BatchNormalization(x)  
x = lbann.Tanh(x)
x = lbann.Deconvolution(x,
						num_dims = 3,
						num_out_channels = number_of_atoms * 2,
						num_groups = 1,
						conv_dims_i = 4,
						conv_pads_i = 1,
						conv_strides_i = 2,
						conv_dilations_i = 1
						has_bias = True
						)
x = lbann.BatchNormalization(x)  
x = lbann.Tanh(x)
x = lbann.Deconvolution(x,
						num_dims = 3,
						num_out_channels = number_of_atoms,
						num_groups = 1,
						conv_dims_i = 4,
						conv_pads_i = 1,
						conv_strides_i = 2,
						conv_dilations_i = 1
						has_bias = True
						)  
decoded = lbann.Tanh(x, 
					name = "decoded")

img_loss = lbann.MeanSquaredError([encoded, tensors]) #Possibly add a sum reduction option
metrics = [lbann.Metric(img_loss, name='recon_error')]

# ----------------------------------
# Set up Experiment
# ----------------------------------

layers = lbann.traverse_layer_graph(input_) #Generate Model DAG

mini_batch_size = args.mini_batch_size
num_epochs = args.num_epochs 

# Callbacks for Debug and Running Model 

print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup

training_output = lbann.CallbackPrint( interval = 1,
										print_global_stat_only = False) #Prints training progress

## Possibly include 3D Voxel Saver Call Back

#Generate Model 
model = lbann.Model(num_epochs,
					layers = layers,
					objective_function = img_loss, 
					metrics = metrics,
					Callbacks = [print_model, training_output]
					)

#Optimizer 

opt = lbann.Adam(1e-6,
				beta1 = 0.9,
				beta2 = 0.99,
				eps = 1e-8
				)

# Data Reader -- TO DO 06/02/2020

data_reader = data.make_data_reader()

#Trainer 

npp = args.ppn # Number of processes (GPUs) per trainer 

trainer = lbann.Trainer(mini_batch_size = mini_batch_size,
						procs_per_trainer = npp,
						name = "MOF_AE_1"
						)

# ----------------------------------
# Run Experiment 
# ----------------------------------

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)

lbann.run(trainer, model, data_reader, opt, **kwargs)