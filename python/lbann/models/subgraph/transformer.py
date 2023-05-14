"""Basic Transformer model with multi-head self-attention.

See:

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention
is all you need." In Advances in Neural Information Processing
Systems, pp. 5998-6008. 2017.

"""
import math
import numpy as np

import lbann
import lbann.modules

class TransformerEncoderLayer(lbann.modules.Module):
	"""Building block for encoder in Transformer model.

	Comprised of multi-head attention and a fully-connected
	feedforward network, each with a residual connection.

	Args:
		embed_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		feedforward_dim (int): Internal dimensionality of
			fully-connected feedforward network.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformerencoderlayer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		embed_dim=512,
		num_heads=8,
		feedforward_dim=2048,
		dropout=0.1,
		d_kv=None,
		name=None,
	):
		TransformerEncoderLayer.global_count += 1
		self.instance = 0
		self.embed_dim = embed_dim
		self.feedforward_dim = feedforward_dim
		self.dropout_prob = dropout

		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformerencoderlayer{TransformerEncoderLayer.global_count}'

		# Layer modules
		self.attention = lbann.modules.subgraph.transformer.MultiheadAttention(
			self.embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention'
		)

		# Weights for fully-connected layers
		self.fc1_weights = [
			lbann.Weights(initializer=lbann.HeNormalInitializer(),
						  name=f'{self.name}_fc1_matrix'),
			lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
						  name=f'{self.name}_fc1_bias'),
		]
		self.fc2_weights = [
			lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
						  name=f'{self.name}_fc2_matrix'),
			lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
						  name=f'{self.name}_fc2_bias'),
		]

	def forward(self, x, mask=None):
		"""Apply Transformer encoder layer.

		Args:
			x (lbann.Layer): Sequence of input vectors.
			mask (lbann.Layer, optional): Attention mask.

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1
		name = f'{self.name}_instance{self.instance}'

		# Self-attention with residual connection
		y = self.attention(x, x, x, mask=mask)
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop1',
			)
		z = lbann.Sum(x, y, name=f'{name}_sum1')
		z = lbann.InstanceNorm(z, name=f'{name}_norm1')
		x = z

		# Feedforward network with residual connection
		y = lbann.ChannelwiseFullyConnected(
			x,
			weights=self.fc1_weights,
			output_channel_dims=[self.feedforward_dim],
			name=f'{name}_fc1',
		)
		y = lbann.Relu(y, name=f'{name}_relu1')
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop2',
			)
		y = lbann.ChannelwiseFullyConnected(
			y,
			weights=self.fc2_weights,
			output_channel_dims=[self.embed_dim],
			name=f'{name}_fc2',
		)
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop3',
			)
		z = lbann.Sum(x, y, name=f'{name}_sum2')
		z = lbann.InstanceNorm(z, name=f'{name}_norm2')
		return z




class TransformerEncoderLayerAllSubgraph(lbann.modules.Module):
	"""Building block for encoder in Transformer model.

	Comprised of multi-head attention and a fully-connected
	feedforward network, each with a residual connection.

	Args:
		embed_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		feedforward_dim (int): Internal dimensionality of
			fully-connected feedforward network.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformerencoderlayer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		embed_dim=512,
		num_heads=8,
		feedforward_dim=2048,
		dropout=0.1,
		d_kv=None,
		name=None,
		Apply_Concat = True,
	):
		TransformerEncoderLayer.global_count += 1
		self.instance = 0
		self.embed_dim = embed_dim
		self.feedforward_dim = feedforward_dim
		self.dropout_prob = dropout
		self.branches = branches
		self.branch_dim  = int( embed_dim // branches)
		self.Apply_Concat = Apply_Concat

		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformerencoderlayer{TransformerEncoderLayer.global_count}'

		# Layer modules
		self.attention = lbann.modules.subgraph.transformer.MultiheadAttentionAllSubGraph(
			self.embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention'
		)

		# Weights for fully-connected layers
		self.fc1_weights = []

		for branch in range(branches):
			self.fc1_weights.append( [
				lbann.Weights(initializer=lbann.HeNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc1_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc1_bias'),
			])

		self.fc2_weights = []

		for branch in range(branches):
			self.fc2_weights.append( [
				lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc2_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc2_bias'),
			])

	def forward(self, x, mask=None):
		"""Apply Transformer encoder layer.

		Args:
			x (lbann.Layer): Sequence of input vectors.
			mask (lbann.Layer, optional): Attention mask.

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1
		name = f'{self.name}_instance{self.instance}'

		# Self-attention with residual connection
		attentions = self.attention(x, x, x, mask=mask)

		slice_points = [self.branch_dim * i for i in range(self.branches+1)]

		x_slice = lbann.Identity(x)

		x_slice = lbann.Slice(
			x_slice,
			axis=1,
			slice_points=slice_points,
			name=f'{name}_x_slice',
			parallel_strategy = {'grid_tag':0}
		)

		head = 0
		branches1 = []
		branches2 = []
		for y in attentions:
			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop1',
				)

			x_sliced = lbann.Identity(x_slice, parallel_strategy = {'grid_tag':head+1})
			z = lbann.Sum(x_sliced, y, name=f'{name}_branch{head}_sum1')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{head}_norm1')
			x = z

			# Feedforward network with residual connection
			y = lbann.ChannelwiseFullyConnected(
				x,
				weights=self.fc1_weights[head],
				output_channel_dims=[self.feedforward_dim],
				name=f'{name}_branch{head}_fc1',
			)

			branches1.append(y)
			branches2.append(x)

			head=head+1


		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branches1)

		branch_outputs = []
		branches1 = []


		for head in range(self.branches):



			y = lbann.Relu(grid_sum_slice, name=f'{name}_branch{head}_relu1', parallel_strategy = {'grid_tag':head+1})
			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop2',
				)
			y = lbann.ChannelwiseFullyConnected(
				y,
				weights=self.fc2_weights[head],
				output_channel_dims=[self.embed_dim],
				name=f'{name}_branch{head}_fc2',
			)
			branches1.append(y)

		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branches1)


		for head in range(self.branches):
			y = lbann.Identity(grid_sum_slice, parallel_strategy = {'grid_tag':head+1})

			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop3',
				)

			z = lbann.Sum(branches2[head], y, name=f'{name}_branch{head}_sum2')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{head}_norm2')

			branch_outputs.append(z)



		if(self.Apply_Concat):
			attentions = lbann.Concatenation(
					branch_outputs,
					axis=1,
					name=f'{name}_heads_concat',parallel_strategy = {'grid_tag':0}
				)


			# Can't have subgraph enabled concat layer just before the split layer for next encoder
			# Problem is in subgrpah parallelism code
			attentions = lbann.Identity(attentions)

		else:
			attentions = branch_outputs
		return attentions



class TransformerEncoderLayerAllSubgraphInputSubGrids(lbann.modules.Module):
	"""Building block for encoder in Transformer model.

	Comprised of multi-head attention and a fully-connected
	feedforward network, each with a residual connection.

	Args:
		embed_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		feedforward_dim (int): Internal dimensionality of
			fully-connected feedforward network.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformerencoderlayer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		embed_dim=512,
		num_heads=8,
		feedforward_dim=2048,
		dropout=0.1,
		d_kv=None,
		name=None,
		Apply_Concat = False
	):
		TransformerEncoderLayer.global_count += 1
		self.instance = 0
		self.embed_dim = embed_dim
		self.feedforward_dim = feedforward_dim
		self.dropout_prob = dropout
		self.branches = branches
		self.branch_dim  = int( embed_dim // branches)
		self.Apply_Concat = Apply_Concat

		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformerencoderlayer{TransformerEncoderLayer.global_count}'

		# Layer modules
		self.attention = lbann.modules.subgraph.transformer.MultiheadAttentionAllSubGraphInputSubGrids(
			self.embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention'
		)

		# Weights for fully-connected layers
		self.fc1_weights = []

		for branch in range(branches):
			self.fc1_weights.append( [
				lbann.Weights(initializer=lbann.HeNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc1_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc1_bias'),
			])

		self.fc2_weights = []

		for branch in range(branches):
			self.fc2_weights.append( [
				lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc2_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc2_bias'),
			])

	def forward(self, x, mask=None):
		"""Apply Transformer encoder layer.

		Args:
			x (lbann.Layer): Sequence of input vectors.
			mask (lbann.Layer, optional): Attention mask.

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1
		name = f'{self.name}_instance{self.instance}'



		# Self-attention with residual connection
		attentions = self.attention(x, x, x, mask=mask)

		slice_points = [self.branch_dim * i for i in range(self.branches+1)]

		x_list = x
		head = 0
		branches1 = []
		branches2 = []
		for iter_count, y in enumerate(attentions):
			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop1',
				)

			x_sliced = lbann.Identity(x_list[iter_count])
			z = lbann.Sum(x_sliced, y, name=f'{name}_branch{head}_sum1')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{head}_norm1')
			x = z

			# Feedforward network with residual connection
			y = lbann.ChannelwiseFullyConnected(
				x,
				weights=self.fc1_weights[head],
				output_channel_dims=[self.feedforward_dim],
				name=f'{name}_branch{head}_fc1',
			)

			branches1.append(y)
			branches2.append(x)

			head=head+1


		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branches1)

		branch_outputs = []
		branches1 = []


		for head in range(self.branches):



			y = lbann.Relu(grid_sum_slice, name=f'{name}_branch{head}_relu1', parallel_strategy = {'grid_tag':head+1})
			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop2',
				)
			y = lbann.ChannelwiseFullyConnected(
				y,
				weights=self.fc2_weights[head],
				output_channel_dims=[self.embed_dim],
				name=f'{name}_branch{head}_fc2',
			)
			branches1.append(y)

		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branches1)


		for head in range(self.branches):
			y = lbann.Identity(grid_sum_slice, parallel_strategy = {'grid_tag':head+1})

			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop3',
				)

			z = lbann.Sum(branches2[head], y, name=f'{name}_branch{head}_sum2')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{head}_norm2')

			branch_outputs.append(z)

		if(self.Apply_Concat):
			attentions = lbann.Concatenation(
					branch_outputs,
					axis=1,
					name=f'{name}_heads_concat',parallel_strategy = {'grid_tag':0}
				)


			# Can't have subgraph enabled concat layer just before the split layer for next encoder
			# Problem is in subgrpah parallelism code
			attentions = lbann.Identity(attentions)

		else:
			attentions = branch_outputs
		return attentions



class TransformerDecoderLayer(lbann.modules.Module):
	"""Building block for decoder in Transformer model.

	Comprised of two multi-head attention modules and a
	fully-connected feedforward network, each with a residual
	connection.

	Args:
		embed_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		feedforward_dim (int): Internal dimensionality of
			fully-connected feedforward network.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformerdecoderlayer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		embed_dim=512,
		num_heads=8,
		feedforward_dim=2048,
		dropout=0.1,
		d_kv = None,
		name=None,
	):
		TransformerDecoderLayer.global_count += 1
		self.instance = 0
		self.embed_dim = embed_dim
		self.feedforward_dim = feedforward_dim
		self.dropout_prob = dropout

		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformerdecoderlayer{TransformerDecoderLayer.global_count}'

		# Layer modules
		self.attention1 = lbann.modules.subgraph.transformer.MultiheadAttention(
			embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention1'
		)
		self.attention2 = lbann.modules.subgraph.transformer.MultiheadAttention(
			embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention2'
		)

		# Weights for fully-connected layers
		self.fc1_weights = [
			lbann.Weights(initializer=lbann.HeNormalInitializer(),
						  name=f'{self.name}_fc1_matrix'),
			lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
						  name=f'{self.name}_fc1_bias'),
		]
		self.fc2_weights = [
			lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
						  name=f'{self.name}_fc2_matrix'),
			lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
						  name=f'{self.name}_fc2_bias'),
		]

	def forward(self, x, memory, src_mask=None, tgt_mask=None):
		"""Apply Transformer decoder layer.

		Args:
			x (lbann.Layer): Sequence of input vectors.
			memory (lbann.Layer): Sequence of vectors produced by
				Transformer encoder stack.
			src_mask (lbann.Layer, optional): Attention mask for
				second attention module (attends to both `x` and
				`memory`).
			tgt_mask (lbann.Layer, optional): Attention mask for first
				attention module (attends only to `x`).

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1
		name = f'{self.name}_instance{self.instance}'

		# Self-attention with residual connection
		y = self.attention1(x, x, x, mask=tgt_mask)
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop1',
			)
		z = lbann.Sum(x, y, name=f'{name}_sum1')
		z = lbann.InstanceNorm(z, name=f'{name}_norm1')
		x = z

		# Attention on encoder output with residual connection
		y = self.attention2(x, memory, memory, mask=src_mask)
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop2',
			)
		z = lbann.Sum(x, y, name=f'{name}_sum2')
		z = lbann.InstanceNorm(z, name=f'{name}_norm2')
		x = z

		# Feedforward network with residual connection
		y = lbann.ChannelwiseFullyConnected(
			x,
			weights=self.fc1_weights,
			output_channel_dims=[self.feedforward_dim],
			name=f'{name}_fc1',
		)
		y = lbann.Relu(y, name=f'{name}_relu1')
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop3',
			)
		y = lbann.ChannelwiseFullyConnected(
			y,
			weights=self.fc2_weights,
			output_channel_dims=[self.embed_dim],
			name=f'{name}_fc2',
		)
		if self.dropout_prob > 0:
			y = lbann.Dropout(
				y,
				keep_prob=1-self.dropout_prob,
				name=f'{name}_drop4',
			)
		z = lbann.Sum(x, y, name=f'{name}_sum3')
		z = lbann.InstanceNorm(z, name=f'{name}_norm3')
		return z


class TransformerDecoderLayerAllSubGraph(lbann.modules.Module):
	"""Building block for decoder in Transformer model.

	Comprised of two multi-head attention modules and a
	fully-connected feedforward network, each with a residual
	connection.

	Args:
		embed_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		feedforward_dim (int): Internal dimensionality of
			fully-connected feedforward network.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformerdecoderlayer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		embed_dim=512,
		num_heads=8,
		feedforward_dim=2048,
		dropout=0.1,
		d_kv = None,
		name=None,
		Apply_Concat=True,
	):
		TransformerDecoderLayer.global_count += 1
		self.instance = 0
		self.embed_dim = embed_dim
		self.feedforward_dim = feedforward_dim
		self.dropout_prob = dropout
		self.Apply_Concat= Apply_Concat
		self.branches = branches
		self.branch_dim  = int( embed_dim // branches)

		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformerdecoderlayer{TransformerDecoderLayer.global_count}'

		# Layer modules
		self.attention1 = lbann.modules.subgraph.transformer.MultiheadAttentionAllSubGraph(
			embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention1'
		)
		self.attention2 = lbann.modules.subgraph.transformer.MultiheadAttentionAllSubGraphInputSubGrids(
			embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention2'
		)

		# Weights for fully-connected layers
		self.fc1_weights = []
		self.fc2_weights = []
		for branch in range(branches):
			self.fc1_weights.append( [
				lbann.Weights(initializer=lbann.HeNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc1_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc1_bias'),
			])
			self.fc2_weights.append( [
				lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc2_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc2_bias'),
			])

	def forward(self, x, memory, src_mask=None, tgt_mask=None):
		"""Apply Transformer decoder layer.

		Args:
			x (lbann.Layer): Sequence of input vectors.
			memory (lbann.Layer): Sequence of vectors produced by
				Transformer encoder stack.
			src_mask (lbann.Layer, optional): Attention mask for
				second attention module (attends to both `x` and
				`memory`).
			tgt_mask (lbann.Layer, optional): Attention mask for first
				attention module (attends only to `x`).

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1
		name = f'{self.name}_instance{self.instance}'


		#Slice x
		x_slice = lbann.Identity(x)

		slice_points = [self.branch_dim * i for i in range(self.branches+1)]

		x_slice = lbann.Slice(
			x_slice,
			axis=1,
			slice_points=slice_points,
			name=f'{name}_x_slice',
			parallel_strategy = {'grid_tag':0}
		)

		memory_slice = lbann.Identity(memory)

		memory_slice = lbann.Slice(
			memory_slice,
			axis=1,
			slice_points=slice_points,
			name=f'{name}_memory_slice',
			parallel_strategy = {'grid_tag':0}
		)

		# Self-attention with residual connection
		y = self.attention1(x, x, x, mask=tgt_mask)


		attentions1_output = []
		memory_list = []

		assert len(y) == self.branches, "Number of layers should be equal to branches "

		for count, attention in enumerate(y):
			if self.dropout_prob > 0:
				attention = lbann.Dropout(
					attention,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{count}_drop1',
				)

			x_sliced = lbann.Identity(x_slice, parallel_strategy = {'grid_tag':count+1})
			x_sliced = lbann.Identity(x_sliced)
			memory_sliced = lbann.Identity(memory_slice, parallel_strategy = {'grid_tag':count+1})
			memory_sliced = lbann.Identity(memory_sliced)
			z = lbann.Sum(x_sliced, attention, name=f'{name}_branch{count}_sum1')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{count}_norm1')
			x = z
			attentions1_output.append(x)
			memory_list.append(memory_sliced)

		# Attention on encoder output with residual connection
		y = self.attention2(attentions1_output, memory_list, memory_list, mask=src_mask)


		branch_outputs1 = []
		branch_outputs2 = []
		for count, attention in enumerate(y):
			if self.dropout_prob > 0:
				attention = lbann.Dropout(
					attention,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{count}_drop2',
				)
			z = lbann.Sum(attentions1_output[count], attention, name=f'{name}_branch{count}_sum2')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{count}_norm2')
			x = z


			branch_outputs1.append(x)
			# Feedforward network with residual connection
			y = lbann.ChannelwiseFullyConnected(
				x,
				weights=self.fc1_weights[count],
				output_channel_dims=[self.feedforward_dim],
				name=f'{name}_branch{count}_fc1',
			)
			branch_outputs2.append(y)


		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branch_outputs2)
		branch_outputs2 = []

		for head in range(self.branches):
			y = lbann.Identity(grid_sum_slice, parallel_strategy = {'grid_tag':head+1})

			y = lbann.Relu(y, name=f'{name}_branch{head}_relu1')
			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop3',
				)
			y = lbann.ChannelwiseFullyConnected(
				y,
				weights=self.fc2_weights[head],
				output_channel_dims=[self.embed_dim],
				name=f'{name}_branch{head}_fc2',
			)
			branch_outputs2.append(y)


		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branch_outputs2)
		branch_outputs2 = []

		for head in range(self.branches):
			y = lbann.Identity(grid_sum_slice, parallel_strategy = {'grid_tag':head+1})

			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop4',
				)
			z = lbann.Sum(branch_outputs1[head], y, name=f'{name}_branch{head}_sum3')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{head}_norm3')
			branch_outputs2.append(z)


		if(self.Apply_Concat):
			attentions = lbann.Concatenation(
					branch_outputs2,
					axis=1,
					name=f'{name}_heads_concat',parallel_strategy = {'grid_tag':0}
				)


			# Can't have subgraph enabled concat layer just before the split layer for next encoder
			# Problem is in subgrpah parallelism code
			attentions = lbann.Identity(attentions)

		else:
			attentions = branch_outputs
		return attentions



class TransformerDecoderLayerAllSubGraphInputSubGrids(lbann.modules.Module):
	"""Building block for decoder in Transformer model.

	Comprised of two multi-head attention modules and a
	fully-connected feedforward network, each with a residual
	connection.

	Args:
		embed_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		feedforward_dim (int): Internal dimensionality of
			fully-connected feedforward network.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformerdecoderlayer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		embed_dim=512,
		num_heads=8,
		feedforward_dim=2048,
		dropout=0.1,
		d_kv = None,
		name=None,
		Apply_Concat=False,
		Slice_X = False
	):
		TransformerDecoderLayer.global_count += 1
		self.instance = 0
		self.embed_dim = embed_dim
		self.feedforward_dim = feedforward_dim
		self.dropout_prob = dropout
		self.Apply_Concat= Apply_Concat
		self.branches = branches
		self.branch_dim  = int( embed_dim // branches)
		self.Slice_X = Slice_X
		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformerdecoderlayer{TransformerDecoderLayer.global_count}'

		# Layer modules
		self.attention1 = lbann.modules.subgraph.transformer.MultiheadAttentionAllSubGraphInputSubGrids(
			embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention1'
		)
		self.attention2 = lbann.modules.subgraph.transformer.MultiheadAttentionAllSubGraphInputSubGrids(
			embed_dim,
			num_heads,
			branches=branches,
			d_kv = d_kv,
			name=f'{self.name}_attention2'
		)

		# Weights for fully-connected layers
		self.fc1_weights = []
		self.fc2_weights = []
		for branch in range(branches):
			self.fc1_weights.append( [
				lbann.Weights(initializer=lbann.HeNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc1_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc1_bias'),
			])
			self.fc2_weights.append( [
				lbann.Weights(initializer=lbann.GlorotNormalInitializer(),
							  name=f'{self.name}_branch{branch}_fc2_matrix'),
				lbann.Weights(initializer=lbann.ConstantInitializer(value=0),
							  name=f'{self.name}_branch{branch}_fc2_bias'),
			])

	def forward(self, x, memory, src_mask=None, tgt_mask=None):
		"""Apply Transformer decoder layer.

		Args:
			x (lbann.Layer): Sequence of input vectors.
			memory (lbann.Layer): Sequence of vectors produced by
				Transformer encoder stack.
			src_mask (lbann.Layer, optional): Attention mask for
				second attention module (attends to both `x` and
				`memory`).
			tgt_mask (lbann.Layer, optional): Attention mask for first
				attention module (attends only to `x`).

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1
		name = f'{self.name}_instance{self.instance}'


		#Slice

		if(self.Slice_X):

			x_slice = lbann.Identity(x)

			slice_points = [self.branch_dim * i for i in range(self.branches+1)]

			x_slice = lbann.Slice(
				x_slice,
				axis=1,
				slice_points=slice_points,
				name=f'{name}_x_slice',
				parallel_strategy = {'grid_tag':0}
			)

			x = []

			for branch in range(self.branches):
				x_sliced = lbann.Identity(x_slice, parallel_strategy = {'grid_tag':branch+1})
				x_sliced = lbann.Identity(x_sliced)
				x.append(x_sliced)


		# Self-attention with residual connection
		y = self.attention1(x, x, x, mask=tgt_mask)


		attentions1_output = []
		memory_list = []
		x_list = x
		memory_list = memory

		assert len(y) == self.branches, "Number of layers should be equal to branches "

		for count, attention in enumerate(y):
			if self.dropout_prob > 0:
				attention = lbann.Dropout(
					attention,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{count}_drop1',
				)

			x_sliced = lbann.Identity(x_list[count])

			z = lbann.Sum(x_sliced, attention, name=f'{name}_branch{count}_sum1')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{count}_norm1')
			x = z
			attentions1_output.append(x)

		# Attention on encoder output with residual connection
		y = self.attention2(attentions1_output, memory_list, memory_list, mask=src_mask)


		branch_outputs1 = []
		branch_outputs2 = []
		for count, attention in enumerate(y):
			if self.dropout_prob > 0:
				attention = lbann.Dropout(
					attention,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{count}_drop2',
				)
			z = lbann.Sum(attentions1_output[count], attention, name=f'{name}_branch{count}_sum2')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{count}_norm2')
			x = z


			branch_outputs1.append(x)
			# Feedforward network with residual connection
			y = lbann.ChannelwiseFullyConnected(
				x,
				weights=self.fc1_weights[count],
				output_channel_dims=[self.feedforward_dim],
				name=f'{name}_branch{count}_fc1',
			)
			branch_outputs2.append(y)


		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branch_outputs2)
		branch_outputs2 = []

		for head in range(self.branches):
			y = lbann.Identity(grid_sum_slice, parallel_strategy = {'grid_tag':head+1})

			y = lbann.Relu(y, name=f'{name}_branch{head}_relu1')
			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop3',
				)
			y = lbann.ChannelwiseFullyConnected(
				y,
				weights=self.fc2_weights[head],
				output_channel_dims=[self.embed_dim],
				name=f'{name}_branch{head}_fc2',
			)
			branch_outputs2.append(y)


		grid_sum_slice = lbann.Cross_Grid_Sum_Slice(branch_outputs2)
		branch_outputs2 = []

		for head in range(self.branches):
			y = lbann.Identity(grid_sum_slice, parallel_strategy = {'grid_tag':head+1})

			if self.dropout_prob > 0:
				y = lbann.Dropout(
					y,
					keep_prob=1-self.dropout_prob,
					name=f'{name}_branch{head}_drop4',
				)
			z = lbann.Sum(branch_outputs1[head], y, name=f'{name}_branch{head}_sum3')
			z = lbann.InstanceNorm(z, name=f'{name}_branch{head}_norm3')
			branch_outputs2.append(z)


		if(self.Apply_Concat):
			attentions = lbann.Concatenation(
					branch_outputs2,
					axis=1,
					name=f'{name}_heads_concat',parallel_strategy = {'grid_tag':0}
				)


			# Can't have subgraph enabled concat layer just before the split layer for next encoder
			# Problem is in subgrpah parallelism code
			attentions = lbann.Identity(attentions)

		else:
			attentions = branch_outputs2
		return attentions


class Transformer(lbann.modules.Module):
	"""Transformer model.

	See:

	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
	Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.
	"Attention is all you need." In Advances in Neural Information
	Processing Systems, pp. 5998-6008. 2017.

	Args:
		hidden_dim (int): Internal dimensionality of multi-head
			attention.
		num_heads (int): Number of attention heads.
		num_encoder_layers (int): Number of stacked layers in encoder.
		num_decoder_layers (int): Number of stacked layers in decoder.
		filter_dim (int): Internal dimensionality of fully-connected
			feedforward networks.
		dropout (float): Dropout probability.
		name (str): Default name is in the form
			'transformer<index>'.

	"""

	global_count = 0  # Static counter, used for default names

	def __init__(
		self,
		branches,
		hidden_size=512,
		num_heads=8,
		num_encoder_layers=6,
		num_decoder_layers=6,
		filter_size=2048,
		dropout=0.1,
		d_kv = None,
		name=None,
		ENABLE_ALLSUBGRAPH=False,
		ENABLE_Concat=False
	):
		Transformer.global_count += 1
		self.instance = 0
		self.hidden_size = hidden_size

		# Apply concat after apply every encoder
		# The input needs to be sliced again among sub-grids
		self.ENABLE_Concat = ENABLE_Concat
		# Module name
		self.name = name
		if not self.name:
			self.name = f'transformer{Transformer.global_count}'

		# Caches for helper functions
		self._subsequent_mask_cache = {}
		self._positional_encoding_cache = {}

		# Encoder and decoder stacks
		if(ENABLE_ALLSUBGRAPH and ENABLE_Concat==False):
			self.encoder = []

			for i in range(num_encoder_layers):
				if(num_encoder_layers == 0):
					self.encoder.append(
					TransformerEncoderLayerAllSubgraph(
						branches,
						embed_dim=hidden_size,
						num_heads=num_heads,
						feedforward_dim=filter_size,
						dropout=dropout,
						d_kv = d_kv,
						name=f'{self.name}_encoder{i}',
						Apply_Concat = False
					))
				elif(i==0 ):
					self.encoder.append(
					TransformerEncoderLayerAllSubgraph(
						branches,
						embed_dim=hidden_size,
						num_heads=num_heads,
						feedforward_dim=filter_size,
						dropout=dropout,
						d_kv = d_kv,
						name=f'{self.name}_encoder{i}',
						Apply_Concat = False
					))
				elif(i == num_encoder_layers-1):
					self.encoder.append(
					TransformerEncoderLayerAllSubgraphInputSubGrids(
						branches,
						embed_dim=hidden_size,
						num_heads=num_heads,
						feedforward_dim=filter_size,
						dropout=dropout,
						d_kv = d_kv,
						name=f'{self.name}_encoder{i}',
						Apply_Concat = False
					))
				else:
					self.encoder.append(
					TransformerEncoderLayerAllSubgraphInputSubGrids(
						branches,
						embed_dim=hidden_size,
						num_heads=num_heads,
						feedforward_dim=filter_size,
						dropout=dropout,
						d_kv = d_kv,
						name=f'{self.name}_encoder{i}',
					))
		elif(ENABLE_ALLSUBGRAPH and ENABLE_Concat==True):
			self.encoder = [
				TransformerEncoderLayerAllSubgraph(
					branches,
					embed_dim=hidden_size,
					num_heads=num_heads,
					feedforward_dim=filter_size,
					dropout=dropout,
					d_kv = d_kv,
					name=f'{self.name}_encoder{i}',
					Apply_Concat = True
				)
				for i in range(num_encoder_layers)]



		else:
			self.encoder = [
				TransformerEncoderLayer(
					branches,
					embed_dim=hidden_size,
					num_heads=num_heads,
					feedforward_dim=filter_size,
					dropout=dropout,
					d_kv = d_kv,
					name=f'{self.name}_encoder{i}',
				)
				for i in range(num_encoder_layers)
			]


		self.decoder = []

		if(ENABLE_ALLSUBGRAPH and ENABLE_Concat ==True):
			self.decoder = [
			TransformerDecoderLayerAllSubGraph(
				branches,
				embed_dim=hidden_size,
				num_heads=num_heads,
				feedforward_dim=filter_size,
				dropout=dropout,
				d_kv = d_kv,
				name=f'{self.name}_decoder{i}',
			)
			for i in range(num_decoder_layers)
			]
		elif (ENABLE_ALLSUBGRAPH and ENABLE_Concat ==False):
			for i in range(num_decoder_layers):
				if(num_decoder_layers==1):
					self.decoder.append(
							TransformerDecoderLayerAllSubGraphInputSubGrids(
								branches,
								embed_dim=hidden_size,
								num_heads=num_heads,
								feedforward_dim=filter_size,
								dropout=dropout,
								d_kv = d_kv,
								name=f'{self.name}_decoder{i}',
								Apply_Concat = True,
								Slice_X = True
							))
				elif(i == num_decoder_layers - 1):
					self.decoder.append(
							TransformerDecoderLayerAllSubGraphInputSubGrids(
								branches,
								embed_dim=hidden_size,
								num_heads=num_heads,
								feedforward_dim=filter_size,
								dropout=dropout,
								d_kv = d_kv,
								name=f'{self.name}_decoder{i}',
								Apply_Concat = True,
								Slice_X = False
							))
				elif(i == 0):
					self.decoder.append(
							TransformerDecoderLayerAllSubGraphInputSubGrids(
								branches,
								embed_dim=hidden_size,
								num_heads=num_heads,
								feedforward_dim=filter_size,
								dropout=dropout,
								d_kv = d_kv,
								name=f'{self.name}_decoder{i}',
								Apply_Concat = False,
								Slice_X = True
							))
				else:
					self.decoder.append(
							TransformerDecoderLayerAllSubGraphInputSubGrids(
								branches,
								embed_dim=hidden_size,
								num_heads=num_heads,
								feedforward_dim=filter_size,
								dropout=dropout,
								d_kv = d_kv,
								name=f'{self.name}_decoder{i}',
								Apply_Concat = False,
								Slice_X = False
							))

		else:
			self.decoder = [
				TransformerDecoderLayer(
					branches,
					embed_dim=hidden_size,
					num_heads=num_heads,
					feedforward_dim=filter_size,
					dropout=dropout,
					d_kv = d_kv,
					name=f'{self.name}_decoder{i}',
				)
				for i in range(num_decoder_layers)
			]
		self.branches = branches

	def _positional_encoding(self, sequence_length):
		"""Positional encodings corresponding to a sequence length.

		PE(pos,2*i)   = sin( pos / 10000**(2*i/hidden_size) )

		PE(pos,2*i+1) = cos( pos / 10000**(2*i/hidden_size) )

		Encodings are memoized.

		"""

		# Construct positional encoding if not in cache
		if sequence_length not in self._positional_encoding_cache:
			vals = []
			for pos in range(sequence_length):
				for i in range((self.hidden_size+1) // 2):
					x = pos / 10000**(2*i/self.hidden_size)
					vals.append(math.sin(x))
					vals.append(math.cos(x))
				if self.hidden_size % 2 != 0:
					vals.pop()
			weights = lbann.Weights(
				initializer=lbann.ValueInitializer(values=vals),
				optimizer=None,
				name=f'{self.name}_positional{sequence_length}_weights',
			)
			self._positional_encoding_cache[sequence_length] = lbann.WeightsLayer(
				dims=[sequence_length, self.hidden_size],
				weights=weights,
				name=f'{self.name}_positional{sequence_length}',
			)

		# Return cached positional encoding
		return self._positional_encoding_cache[sequence_length]

	def _subsequent_mask(self, size):
		"""Attention mask to prevent attending to subsequent positions.

		The (i,j) entry is -1e9 if i<j and is 0 otherwise. Masks are
		memoized.

		"""

		# Construct mask if not in cache
		if size not in self._subsequent_mask_cache:
			vals = np.triu(np.full((size,size), -1e9), k=1)
			weights = lbann.Weights(
				initializer=lbann.ValueInitializer(values=vals.flat),
				optimizer=None,
				name=f'{self.name}_mask{size}_weights',
			)
			self._subsequent_mask_cache[size] = lbann.WeightsLayer(
				dims=[size, size],
				weights=weights,
				name=f'{self.name}_mask{size}',
			)

		# Return cached mask
		return self._subsequent_mask_cache[size]

	def forward(self, source, source_length, target, target_length):
		"""Apply Transformer.

		The input and output tensors are interpreted as sequences of
		vectors, where the first tensor dimension is the sequence
		dimension.

		Args:
			source (lbann.Layer): Sequence of input vectors to encoder
				stack.
			source_length (int): Length of input sequence to encoder.
			target (lbann.Layer): Sequence of input vectors to decoder
				stack.
			target_length (int): Length of input sequence to decoder.

		Returns:
			lbann.Layer: Sequence of output vectors.

		"""
		self.instance += 1

		# Encoder stack
		# Note: Add positional encoding to input
		x = lbann.Sum([
			source,
			self._positional_encoding(source_length)],
			name=f'{self.name}_instance{self.instance}_positional_source',
		)
		for encoder_layer in self.encoder:
			x = encoder_layer(x)
		memory = x

		# Decoder stack
		# Note: Add positional encoding to input
		x = lbann.Sum(
			[target,
			self._positional_encoding(target_length)],
			name=f'{self.name}_instance{self.instance}_positional_target',
		)

		subgraph_masks = {}


		if(self.branches>0):
			for i in range(self.branches):
				subgraph_masks[i+1] = lbann.Identity(self._subsequent_mask(target_length),name="mylayer"+str(i) ,
									parallel_strategy = {'grid_tag':i+1})
				subgraph_masks[i+1] = lbann.Identity(subgraph_masks[i+1])


		if(self.branches>0):
			for decoder_layer in self.decoder:
				x = decoder_layer(
					x,
					memory,
					tgt_mask=subgraph_masks,
				)

		else:
			for decoder_layer in self.decoder:
				x = decoder_layer(
					x,
					memory,
					tgt_mask=self._subsequent_mask(target_length),
				)

		return x
