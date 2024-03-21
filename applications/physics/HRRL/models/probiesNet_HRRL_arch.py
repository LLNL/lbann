import torch.nn as nn

class PROBIESNet(nn.Module):

	global_count = 0  # Static counter, used for default names

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, stride=1, kernel_size=3, padding=0, dilation=1)
		self.maxPool1 = nn.MaxPool2d(kernel_size=2,padding=0,dilation=1)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=1, kernel_size=3, padding=0, dilation=1)
		self.avgPool1 = nn.AvgPool2d(kernel_size=2,stride=2,padding=1)
		self.flatten1 = nn.Flatten(start_dim=1, end_dim=-1)
		self.leakyReLu1 = nn.LeakyReLU()
		self.dense1 = nn.Linear(in_features=350464,out_features=128)
		self.leakyReLu2 = nn.LeakyReLU()
		self.dense2 = nn.Linear(in_features=128,out_features=2048)
		# double-check the out_features when making changes.
		self.dense3 = nn.Linear(in_features=2048, out_features=5) 


	def forward(self, x):
		x = self.conv1(x)
		x = self.maxPool1(x)
		x = self.conv2(x)
		x = self.avgPool1(x)
		x = self.flatten1(x)
		x = self.leakyReLu1(x)
		x = self.dense1(x)
		x = self.leakyReLu2(x)
		x = self.dense2(x)
		x = self.dense3(x)
		return x
