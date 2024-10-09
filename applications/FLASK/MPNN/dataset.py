import pickle
import numpy as np


MAX_ATOMS = 100  # The number of maximum atoms in CSD dataset
MAX_BONDS = 224  # The number of maximum bonds in CSD dataset
ATOM_FEATURES = 133
BOND_FEATURES = 147

SAMPLE_SIZE = (
    (MAX_ATOMS * ATOM_FEATURES)
    + (MAX_BONDS * BOND_FEATURES)
    + 4 * MAX_BONDS
    + MAX_ATOMS
    + 2
)

DATA_DIR = "/p/vast1/lbann/datasets/FLASK/CSD10K/"

with open(DATA_DIR + "10k_density_lbann.bin", "rb") as f:
    data = pickle.load(f)

train_index = np.load(DATA_DIR + "train_sample_indices.npy")
valid_index = np.load(DATA_DIR + "valid_sample_indices.npy")
test_index = np.load(DATA_DIR + "test_sample_indices.npy")


def padded_index_array(size, special_ignore_index=-1):
    padded_indices = np.zeros(size, dtype=np.float32) + special_ignore_index
    return padded_indices


def pad_data_sample(data):
    """
    Args:
        data(dict): Dictionary of data samples with fields 'num_atoms', 'num_bonds',
            'dual_graph_atom2bond_source', 'dual_graph_atom2bond_target',
            'bond_graph_source', 'bond_grap_target', and 'target'

    Returns:
        (np.array) 
    """
    num_atoms = data["num_atoms"]
    num_bonds = data["num_bonds"]
    f_atoms = np.zeros((MAX_ATOMS, ATOM_FEATURES), dtype=np.float32)
    f_atoms[:num_atoms, :] = data["atom_features"]

    f_bonds = np.zeros((MAX_BONDS, BOND_FEATURES), dtype=np.float32)

    f_bonds[:num_bonds, :] = data["bond_features"]

    atom2bond_source = padded_index_array(MAX_BONDS)
    atom2bond_source[:num_bonds] = data["dual_graph_atom2bond_source"]

    atom2bond_target = padded_index_array(MAX_BONDS)
    atom2bond_target[:num_bonds] = data["dual_graph_atom2bond_target"]

    bond2atom_source = padded_index_array(MAX_BONDS)
    bond2atom_source[:num_bonds] = data["bond_graph_source"]
    bond2bond_target = padded_index_array(MAX_BONDS)
    bond2bond_target[:num_bonds] = data["bond_graph_target"]

    atom_mask = padded_index_array(MAX_ATOMS)
    atom_mask[:num_atoms] = np.zeros(num_atoms)

    num_atoms = np.array([num_atoms]).astype(np.float32)
    target = (np.array([data["target"]]).astype(np.float32) + 67.14776709141553) / (
        108.13423283538837
    )

    _data_array = [
        f_atoms.flatten(),
        f_bonds.flatten(),
        atom2bond_source.flatten(),
        atom2bond_target.flatten(),
        bond2atom_source.flatten(),
        bond2bond_target.flatten(),
        atom_mask.flatten(),
        num_atoms.flatten(),
        target.flatten(),
    ]

    flattened_data_array = np.concatenate(_data_array, axis=None)
    return flattened_data_array


def train_sample(index):
    return pad_data_sample(data[train_index[index]])


def validation_sample(index):
    return pad_data_sample(data[valid_index[index]])


def test_sample(index):
    return pad_data_sample(data[test_index[index]])


def train_num_samples():
    return 8164


def validation_num_samples():
    return 1020


def test_num_samples():
    return 1022


def sample_dims():
    return (SAMPLE_SIZE,)


if __name__ == "__main__":
    print(train_sample(2).shape, sample_dims())
