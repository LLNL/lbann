import numpy as np 
from util import GRID_Parser
from util import INPtoTensor

		
class Voxel_MOF(object):
	"""docstring for Voxel_MOF"""
	def __init__(self, inp_file_path, grid_file_path):
		super(Voxel_MOF, self).__init__()
		grid = GRID_Parser.GRID_Parser(grid_file_path)
		pos = INPtoTensor.INPtoTensor(inp_file_path)
		self.grid_tensor = grid.get_grid()
		self.loc_tensor = pos.get_Tensor()
		self.lattice_params = pos.get_lattice_params()
		self.grid_metadata = grid.get_metadata()
		self.data = np.append(self.loc_tensor, np.expand_dims(self.grid_tensor, axis=0), axis=0)
	def get_data(self):
		return self.data

	def get_lattice_params(self):
		return self.lattice_params


def main():
	MOF = Voxel_MOF("AHOKOX_clean.inp","AHOKOX_clean.grid")
	X = MOF.get_data()
	print(X.shape)
	print(MOF.get_lattice_params())
if __name__ == '__main__':
	main()