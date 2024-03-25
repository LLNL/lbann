from config import DATASET_CONFIG
from tqdm import tqdm
import numpy as np
from chemprop.args import TrainArgs
from chemprop.features import reset_featurization_parameters
from chemprop.data import MoleculeDataLoader, utils
import os.path as osp
import pickle


def retrieve_dual_mapping(atom2bond, ascope):
    atom_bond_mapping = []
    for a_start, a_size in enumerate:
        _a2b = atom2bond.narrow(0, a_start, a_size)
        for row, possible_bonds in enumerate(_a2b):
            for bond in possible_bonds:
                ind = bond.item() - 1  # Shift by 1 to account for null nodes
                if ind >= 0:
                    atom_bond_mapping.append([row, ind])
    return np.array(atom_bond_mapping)


def PrepareDataset(save_file_name, target_file):
    data_file = osp.join(DATASET_CONFIG["DATA_DIR"], target_file)

    arguments = [
        "--data_path",
        data_file,
        "--dataset_type",
        "regression",
        "--save_dir",
        "./data/10k_dft_density",
    ]
    args = TrainArgs().parse_args(arguments)
    reset_featurization_parameters()
    data = utils.get_data(data_file, args=args)
    #  Need to use the data loader as the featurization happens in the dataloader
    #  Only use 1 mol as in LBANN we do not do coalesced batching (yet)
    dataloader = MoleculeDataLoader(data, batch_size=1)
    lbann_data = []
    for mol in tqdm(dataloader):
        mol_data = {}

        mol_data["target"] = mol.targets()[0][0]
        mol_data["num_atoms"] = mol.number_of_atoms[0][0]
        # Multiply by 2 for directional bonds
        mol_data["num_bonds"] = mol.number_of_bonds[0][0] * 2

        mol_batch = mol.batch_graph()[0]
        f_atoms, f_bonds, a2b, b2a, b2revb, ascope, bscope = mol_batch.get_components(
            False
        )

        # shift by 1 as we don't use null nodes as in the ChemProp implementation
        mol_data["atom_features"] = f_atoms[1:].numpy()
        mol_data["bond_features"] = f_bonds[1:].numpy()
        dual_graph_mapping = retrieve_dual_mapping(a2b, ascope)

        mol_data['dual_graph_atom2bond_source'] = dual_graph_mapping[:, 0]
        mol_data['dual_graph_atom2bond_target'] = dual_graph_mapping[:, 1]

        # subtract 1 to shift the indices
        mol_data['bond_graph_source'] = b2a[1:].numpy() - 1
        mol_data['bond_graph_target'] = b2revb[1:].numpy() - 1

        lbann_data.append(mol_data)

    save_file = osp.join(DATASET_CONFIG["DATA_DIR"], save_file_name)
    with open(save_file, 'wb') as f:
        pickle.dump(lbann_data, f)


def main():
    PrepareDataset("10k_density_lbann.bin", "10k_dft_density_data.csv")
    PrepareDataset("10k_hof_lbann.bin", "10k_dft_hof_data.csv")


if __name__ == "__main__":
    main()
