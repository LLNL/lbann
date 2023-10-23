import lbann
from config import DATASET_CONFIG, HYPERPARAMETERS_CONFIG
from MPN import MPNEncoder


def graph_splitter(_input):
    """
    """
    split_indices = [0]
  

    max_atoms = DATASET_CONFIG['MAX_ATOMS'],
    max_bonds = DATASET_CONFIG['MAX_BONDS'],
    atom_features = DATASET_CONFIG['ATOM_FEATURES']
    bond_features = DATASET_CONFIG['BOND_FEATURES']

    indices_length = max_bonds

    f_atom_size = max_atoms * atom_features
    split_indices.append(f_atom_size)
    
    f_bond_size = max_bonds * bond_features
    split_indices.append(f_bond_size)
  
    split_indices.append(max_bonds)
    split_indices.append(max_bonds)
    split_indices.append(max_bonds)
    split_indices.append(max_bonds)

    split_indices.append(max_atoms)
    split_indices.append(1)
    split_indices.append(1)


    for i in range(1, len(split_indices)):
        split_indices[i] = split_indices[i] + split_indices[i - 1]    
    
    graph_input = lbann.Slice(_input, axis=0, slice_points=split_indices)
    f_atoms = lbann.Reshape(lbann.Identity(graph_input), dims=[max_atoms, atom_features])
    f_bonds = lbann.Reshape(lbann.Identity(graph_input), dims=[max_bonds, bond_features])
    atom2bond_source_mapping = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[max_bonds])
    atom2bond_target_mapping = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[max_bonds])
    bond2atom_mapping = lbann.Reshape(lbann.Identity(graph_input), dims=[max_bonds])
    bond2bond_mapping = lbann.Reshape(lbann.Identity(graph_input), dims=[max_bonds])
    graph_mask = lbann.Reshape(lbann.Identity(graph_input), dims=[max_atoms])
    num_atoms = lbann.Reshape(lbann.Identity(graph_input), dims=[1])
    target = lbann.Reshape(lbann.Identity(graph_input), dims=[1])
    
    return f_atoms, f_bonds, atom2bond_source_mapping, atom2bond_target_mapping, \
      bond2atom_mapping, bond2bond_mapping, graph_mask, num_atoms, target


def make_model():
    _input = lbann.Input(data_field='samples')

    f_atoms, f_bonds, atom2bond_source_mapping, atom2bond_target_mapping, \
      bond2atom_mapping, bond2bond_mapping, graph_mask, num_atoms, target = graph_splitter(_input)

    encoder = MPNEncoder(atom_fdim=DATASET_CONFIG['ATOM_FEATURES'],
                         bond_fdim=DATASET_CONFIG['BOND_FEATURES'],
                         hidden_size=HYPERPARAMETERS_CONFIG['HIDDEN_SIZE'],
                         activation_func=lbann.Relu)

    encoded_vec = encoder(f_atoms,
                          f_bonds,
                          atom2bond_source_mapping,
                          atom2bond_target_mapping,
                          bond2atom_mapping,
                          bond2bond_mapping,
                          graph_mask,
                          num_atoms)

    # Readout layers
    x = lbann.FullyConnected(encoded_vec, num_neurons=HYPERPARAMETERS_CONFIG['HIDDEN_SIZE'],
                             name="READOUT_Linear_1")
    x = lbann.Relu(x, name="READOUT_Activation_1")

    x = lbann.FullyConnected(x, num_neurons=1,
                             name="READOUT_output")

    loss = lbann.MeanSquaredError(x, target)

    layers = lbann.traverse_layer_graph(_input)
    training_output = lbann.CallbackPrint(interval=1, print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()

    callbacks = [training_output, gpu_usage, timer]
    model = lbann.Model(HYPERPARAMETERS_CONFIG['EPOCH'],
                        layers=layers,
                        objective_function=loss,
                        callbacks=callbacks)
    return model

