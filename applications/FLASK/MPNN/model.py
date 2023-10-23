import lbann
from config import DATASET_CONFIG
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

    for i in range(1, len(split_indices)):
        split_indices[i] = split_indices[i] + split_indices[i - 1]    
    
    graph_input = lbann.Slice(_input, axis=0, slice_points=split_indices)
    f_atoms = lbann.Reshape(lbann.Identity(graph_input), dims=[max_atoms, atom_features])
    f_bonds = lbann.Reshape(lbann.Identity(graph_input), dims=[max_bonds, bond_features])
    atom2bond_source_mapping = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[max_bonds])
    atom2bond_target_mapping = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[max_bonds])
    bond2atom_mapping = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[max_bonds])
    bond2bond_mapping = lbann.Reshape(lbann.Identity(graph_input),
                                      dims=[max_bonds])
    graph_mask = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[max_atoms])
    num_atoms = lbann.Reshape(lbann.Identity(graph_input),
                                             dims=[1])
    
    return f_atoms, f_bonds, atom2bond_source_mapping, atom2bond_target_mapping, \
      bond2atom_mapping, bond2bond_mapping, graph_mask, num_atoms


def make_model():


