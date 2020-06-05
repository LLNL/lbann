import numpy as np

class GRID_Parser(object):
	"""
	Class energy grids generated from CIF files
	"""
	def __init__(self, grid_file_path, grid_dimensions=(32,32,32)):
		super(GRID_Parser, self).__init__()
		
		f = open(grid_file_path,'r')
		f_data = f.readlines()
		f.close()

		self.grid_tensor = self.parse_tensor(f_data,grid_dimensions)
		self.metadata = self.parse_metadata(f_data)


		
	def get_grid(self):
		return self.grid_tensor
	def get_metadata(self):
		return self.metadata 
	def parse_metadata(self, file_data):
		dic = {}
		dic['Grid File Name'] = file_data[0].strip().split()[-1]
		dic['Creation Time'] = file_data[1].strip()
		dic['Version'] = file_data[2].strip().split()[-1]
		dic['Input File Name'] = file_data[3].strip().split()[-1]
		dic['Lattice Vectors'] = [float(x) for x in file_data[8].strip().split()]
		dic['Lattice Angles'] = [float(x) for x in file_data[10].strip().split()]
		calculation_param_names = file_data[11].strip().split()
		calculation_params = file_data[12].strip().split()

		for i in range(len(calculation_param_names)):
			dic[calculation_param_names[i]] = float(calculation_params[i])


		return dic
	def parse_tensor(self, f_data, grid_dimensions):

		grid_tensor = np.zeros(grid_dimensions)
		if (len(f_data[15:]) != grid_dimensions[0]*grid_dimensions[1]*grid_dimensions[2]):
			print("Invalid grid dimensions ")
			raise ValueError

		line_num = 15 
		for x in range(grid_dimensions[0]):
			for y in range(grid_dimensions[1]):
				for z in range(grid_dimensions[2]):

					line = f_data[line_num]
					val = float(line.strip().split()[-1])
					grid_tensor[x][y][z] = val
					line_num +=1
		return grid_tensor