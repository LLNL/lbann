import numpy as np
from .Atom import Atom 

class INPtoTensor(object):
	"""

	Custom dataset for transforming input file INP to Tensors with location data

	"""
	def __init__(self, inp_file_path):
		super(INPtoTensor, self).__init__()
		inp_file = open(inp_file_path,'r')
		file_data = inp_file.readlines()
		inp_file.close()


		self.atoms = []
		self.a,self.b,self.c = self.__parse_lattice_vectors(file_data)
		self.alpha, self.beta, self.gamma = self.__parse_lattice_angles(file_data)

		self.__parse_atom_details(file_data)
	def __parse_lattice_vectors(self, file_str):
		vector_data = file_str[9].strip().split()

		
		a = float(vector_data[0])
		b = float(vector_data[1]) 
		c = float(vector_data[2]) 

		return a,b,c
	def __parse_lattice_angles(self, file_str):
		angle_data = file_str[11].strip().split()

		alpha = float(angle_data[0])
		beta = float(angle_data[1])
		gamma = float(angle_data[2])
		return np.deg2rad(alpha),np.deg2rad(beta),np.deg2rad(gamma)

	def __parse_atom_details(self, file_str):
		
		for atom in file_str[17:]:
			atom = atom.strip().split()
			frac_a = float(atom[-4])
			frac_b = float(atom[-3])
			frac_c = float(atom[-2])
			x,y,z = self.frac2car(frac_a,frac_b,frac_c)
			specie = atom[-1]
			self.atoms.append(Atom(specie,x,y,z))


	def frac2car(self, frac_a, frac_b, frac_c):
		cos_alpha = np.cos(self.alpha)
		cos_beta = np.cos(self.beta)
		cos_gamma = np.cos(self.gamma)

		sin_alpha = np.sin(self.alpha)
		sin_beta = np.sin(self.beta)
		sin_gamma = np.sin(self.gamma)

		a = self.a
		b = self.b
		c = self.c


		sigma_c = (sin_gamma ** 2) - (sin_gamma**2 * cos_gamma**2) - (cos_alpha - cos_gamma*cos_beta)**2
		OMEGA = a * b * c * (np.sqrt(sigma_c))
		x = a * frac_a + (b * cos_gamma)*frac_b + (c * cos_beta)*frac_c
		y = (b * sin_gamma) * frac_b + (c * (cos_alpha - (cos_beta * cos_gamma)) / sin_gamma)*frac_c
		z = (OMEGA / (a * b * sin_gamma)) * frac_c
		return x,y,z

	def __add_mol_gaussian(self, tensor, specie, x,y,z, variance=0.5):
		shape = tensor[specie].shape
		distances = np.zeros(shape)

		for x_i in range(shape[0]):
			for y_i in range(shape[1]):
				for z_i in range(shape[2]):
					x_i_val,y_i_val,z_i_val = self.frac2car((x_i / shape[0]),(y_i / shape[1]),(z_i / shape[2])) 
					distances[x_i][y_i][z_i] = (-0.5)*((x_i_val - x)**2 + (y_i_val - y)**2 +(z_i_val-z)**2)/(variance**2)
		distances = np.exp(distances)
		distances = np.power(2/np.pi,3/2) * distances
		assert distances.shape == shape 
		tensor[specie] += distances
		
		return tensor[specie]
	def get_Tensor(self,
		atom_species=[],
		dimensions=(32,32,32),
		spread=0.5):

		if(len(atom_species) ==0):
			# print('Atom Species not specified. Using default_atoms:  ["H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"] ')
			atom_species = ["H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"]

		dimensions = (len(atom_species), *dimensions)
		mol_tensor = np.zeros(dimensions)

		for atom in self.atoms:
			specie = atom_species.index(atom.specie)
			x,y,z = atom.get_pos()
			mol_tensor[specie] = self.__add_mol_gaussian(mol_tensor,specie,x,y,z,variance=spread)
		return mol_tensor	

	def get_lattice_params(self):
		dic = {}

		dic['a'] = self.a
		dic['b'] = self.b 
		dic['c'] = self.c

		dic['alpha'] = self.alpha
		dic['beta'] = self.beta
		dic['gamma'] = self.gamma

		return dic