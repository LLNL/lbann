class Atom(object):
	"""Atom object for use in CIF to Tensor transformation"""
	def __init__(self, specie, x,y,z):
		super(Atom, self).__init__()
		self.specie = specie
		self.x = x
		self.y = y
		self.z = z

	def get_pos(self):
		return self.x,self.y,self.z