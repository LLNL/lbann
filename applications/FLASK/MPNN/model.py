import lbann
from config import DATASET_CONFIG, HYPERPARAMETERS_CONFIG
from MPN import MPNEncoder
import os.path as osp


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
    target = lbann.Reshape(lbann.Identity(graph_input), dims=[1], name='TARGET')
    
    return f_atoms, f_bonds, atom2bond_source_mapping, atom2bond_target_mapping, \
      bond2atom_mapping, bond2bond_mapping, graph_mask, num_atoms, target


def make_model():
    _input = lbann.Input(data_field='samples')

    f_atoms, f_bonds, atom2bond_source_mapping, atom2bond_target_mapping, \
      bond2atom_mapping, bond2bond_mapping, graph_mask, num_atoms, target = graph_splitter(_input)

    encoder = MPNEncoder(atom_fdim=DATASET_CONFIG['ATOM_FEATURES'],
                         bond_fdim=DATASET_CONFIG['BOND_FEATURES'],
                         max_atoms=DATASET_CONFIG['MAX_ATOMS'],
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
                             name="PREDICTION")

    loss = lbann.MeanSquaredError(x, target)

    layers = lbann.traverse_layer_graph(_input)

    # Callbacks
    training_output = lbann.CallbackPrint(interval=1, print_global_stat_only=False)
    gpu_usage = lbann.CallbackGPUMemoryUsage()
    timer = lbann.CallbackTimer()
    predictions = lbann.CallbackDumpOutputs(['TARGET', 'PREDICTION'],
                                            role='test')

    callbacks = [training_output, gpu_usage, timer, predictions]
    model = lbann.Model(HYPERPARAMETERS_CONFIG['EPOCH'],
                        layers=layers,
                        objective_function=loss,
                        callbacks=callbacks)
    return model


def make_data_reader(classname='dataset',
                     sample='get_sample_func',
                     num_samples='num_samples_func',
                     sample_dims='sample_dims_func'):
    data_dir = osp.dirname(osp.realpath(__file__))
    reader = lbann.reader_pb2.DataReader()

    for role in ['train', 'validation', 'test']:
        _reader = reader.reader.add()
        _reader.name = 'python'
        _reader.role = role
        _reader.shuffle = True
        _reader.fraction_of_data_to_use = 1.0
        _reader.python.module = classname
        _reader.python.module_dir = data_dir
        _reader.python.sample_function = f"{role}_{sample}"
        _reader.python.num_samples_function = f"{role}_{num_samples}"
        _reader.python.sample_dims_function = f"{role}_{sample_dims}"
    
    return reader