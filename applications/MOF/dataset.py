import os 
import numpy as np 
from data.MOFdataset import MOFDataset 

# MOFdaset is a custom dataset class extending torch.utils.data.Dataset

##
## For an example look at: 
## https://github.com/LLNL/lbann/blob/develop/applications/nlp/transformer/dataset.py
##

data_dir = os.path.dirname(os.path.realpath(__file__))

## Add CLI arguments for training file location and error handling 
train_file_path = os.path.join(data_dir, 'data/mofs.p')  
test_file_path = os.path.join(data_dir, 'data/mofs.p')

training_data = MOFDataset(train_file_path, no_grid=True)
test_data = MOFDataset(test_file_path, no_grid=True)

def get_train (index):
	return np.float32(training_data[index].flatten()) #Iterable or 1 D array 

def get_test (index):
	return np.float32(test_data[index].flatten()) #Iterable or 1D array
def num_train_samples():
	return len(training_data)

def num_test_samples():
	return len(test_data)

def sample_dims():
	return (32*32*32*11, )

if __name__ == '__main__':
	data_dir = os.path.dirname(os.path.realpath(__file__))

## Add CLI arguments for training file location and error handling 
	train_file_path = os.path.join(data_dir, 'data/mofs.p')  
	test_file_path = os.path.join(data_dir, 'data/mofs.p')

	training_data =MOFDataset(train_file_path, no_grid=True)
	test_data = MOFDataset(test_file_path, no_grid=True)

	print(len(training_data))
	print(training_data[0].shape)


	
