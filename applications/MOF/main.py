import argparse 
import lbann
import MOFae
import dataset
import os 
import lbann.contrib.launcher
import lbann.contrib.args
# ----------------------------------
# Command-line arguments
# ----------------------------------


desc = ("Training 3D-CAE on 4D MOF Data using LBANN")

parser = argparse.ArgumentParser(description = desc)

parser.add_argument(
	'--zdim', action='store',default = 2048, type=int, 
	help="dimensionality of latent space (dedfault: 2048)", metavar = 'NUM')
parser.add_argument(
	'--atoms', action='store', default = 11,type=int, 
	help="Number of atom species (default: 11)", metavar = 'NUM')
parser.add_argument(
    '--mini-batch-size', action='store', default=128, type=int,
    help='mini-batch size (default: 128)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=100, type=int,
    help='number of epochs (default: 100)', metavar='NUM')

lbann.contrib.args.add_scheduler_arguments(parser, 'mofae')
args = parser.parse_args()


latent_dim = args.zdim
number_of_atoms = args.atoms


layers, img_loss, metrics = MOFae.gen_layers(latent_dim, number_of_atoms)
mini_batch_size = args.mini_batch_size
num_epochs = args.num_epochs 

# Callbacks for Debug and Running Model 

print_model = lbann.CallbackPrintModelDescription() #Prints initial Model after Setup

training_output = lbann.CallbackPrint( interval = 1,
				       print_global_stat_only = False) #Prints training progress

gpu_usage = lbann.CallbackGPUMemoryUsage()

encoded_output = lbann.CallbackDumpOutputs( layers = "decoded", batch_interval = 400, directory = os.path.dirname(os.path.realpath(__file__)), format="npy") 

# ----------------------------------
# Set up Experiment
# ----------------------------------

#Generate Model 
model = lbann.Model(num_epochs,
		    layers = layers,
		    objective_function = img_loss, 
		    metrics = metrics,
		    callbacks = [print_model, training_output, gpu_usage, encoded_output]
		   )

#Optimizer 

opt = lbann.Adam(learn_rate = 1e-2,
		beta1 = 0.9,
		beta2 = 0.99,
		eps = 1e-8
	        )

data_reader = MOFae.make_data_reader()


#Trainer 

trainer = lbann.Trainer(mini_batch_size = mini_batch_size,
						name = "MOF_AE_1"
						)

# ----------------------------------
# Run Experiment 
# ----------------------------------

kwargs = lbann.contrib.args.get_scheduler_kwargs(args)


lbann.contrib.launcher.run(trainer, model, data_reader, opt, **kwargs)
