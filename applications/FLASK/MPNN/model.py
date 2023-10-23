import lbann
from config import DATASET_CONFIG
from MPN import MPNEncoder


def graph_splitter(_input):
    """
    """
    split_indices = []
    start_index = 0

    max_atoms = DATASET_CONFIG['MAX_ATOMS'],
    max_bonds = DATASET_CONFIG['MAX_BONDS'],
    atom_features = DATASET_CONFIG['ATOM_FEATURES']
    bond_features = DATASET_CONFIG['BOND_FEATURES']

    f_atom_size = max_atoms * atom_features
    f_bond_size = max_bonds * bond_features

    


    return f_atoms, f_bonds, atom2bond_mapping, bond2atom_mapping,\
      bond2bond_mapping, graph_mask, num_atoms


def make_model():


