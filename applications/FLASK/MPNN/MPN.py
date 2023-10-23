import lbann
from lbann.modules import Module, ChannelwiseFullyConnectedModule


class MPNEncoder(Module):
  """
  """
  global_count = 0

  def __init__(self, atom_fdim, bond_fdim, hidden_size, activation_func, bias=False, depth=3, name=None):
    
      MPNEncoder.global_count += 1
      # For debugging
      self.name = (name
                  if name
                  else 'MPNEncoder_{}'.format(MPNEncoder.global_count))
      
      self.atom_fdim = atom_fdim
      self.bond_fdim = bond_fdim
      self.hidden_size = hidden_size
      self.bias = bias
      self.depth = depth
      self.activation_func = activation_func

      # Channelwise fully connected layer: (*, *, bond_fdim) -> (*, *, hidden_size)
      self.W_i = \
          ChannelwiseFullyConnectedModule(self.hidden_size,
                                          bias=self.bias,
                                          activation=self.activation_func,
                                          name=self.name + "W_i")

      # Channelwise fully connected layer (*, *, hidden_size) -> (*, *, hidden_size))
      self.W_h = \
          ChannelwiseFullyConnectedModule(self.hidden_size,
                                          bias=self.bias,
                                          activation=self.activation_func,
                                          name=self.name + "W_h")
      # Channelwise fully connected layer (*, *, atom_fdim + hidden_size) -> (*, *, hidden_size))
      self.W_o = \
          ChannelwiseFullyConnectedModule(self.hidden_size,
                                          bias=True,
                                          activation=self.activation_func,
                                          name=self.name + "W_o")

  
  def message(self, bond_features,
              bond2atom_mapping,
              atom2bond_sources_mapping,
              atom2bond_target_mapping,
              bond2revbond_mapping,
              MAX_ATOMS):
      """
      """
      messages = self.W_i(bond_features)
      for depth in range(self.depth - 1):
          nei_message = lbann.Gather(messages,
                                     atom2bond_sources_mapping,
                                     axis=0)

          a_message = lbann.Scatter(nei_message,
                                    atom2bond_target_mapping,
                                    dims=[MAX_ATOMS, self.hidden_size],
                                    axis=0)

          bond_message = lbann.Gather(a_message,
                                      bond2atom_mapping)
          rev_message = lbann.Gather(messages,
                                     bond2revbond_mapping)

          messages = lbann.SubtractOperator(bond_message, rev_message)
          messages = self.W_h(messages)

      return messages
  

  def aggregate(self,
                atom_messages,
                bond_messages,
                bond2atom_mapping,
                NUM_ATOMS):
      """
      """
      a_messages = lbann.Scatter(bond_messages,
                                 bond2atom_mapping,
                                 axis=0,
                                 dims=[NUM_ATOMS, self.hidden_size])

      atoms_hidden = lbann.Concatentate([atom_messages, a_messages],
                                        dim=0)
      return self.W_o(atoms_hidden)


  def readout(self,
              atom_encoded_features,
              graph_mask,
              num_atoms,
              max_atoms):
      """
      """
      mol_encoding = lbann.Scatter(atom_encoded_features,
                                   graph_mask,
                                   name=self.name + "graph_scatter")
      mol_encoding = lbann.DivideOperator(mol_encoding,
                                          lbann.Tessallate(num_atoms,
                                                           dims=[max_atoms, 1]))
      return mol_encoding


  def forward(self,
              atom_input_features,
              bond_input_features,
              atom2bond_sources_mapping,
              atom2bond_target_mapping,
              bond2atom_mapping,
              bond2revbond_mapping,
              graph_mask, num_atoms, max_atoms):
      """
      """
      bond_messages = self.message(bond_input_features,
                                   bond2atom_mapping,
                                   atom2bond_sources_mapping,
                                   atom2bond_target_mapping,
                                   bond2revbond_mapping,
                                   max_atoms)
      
      atom_encoded_features = self.aggregate(atom_input_features,
                                             bond_messages,
                                             bond2atom_mapping,
                                             num_atoms)
      
      readout = self.readout(atom_encoded_features, graph_mask, num_atoms, max_atoms)
      return readout
