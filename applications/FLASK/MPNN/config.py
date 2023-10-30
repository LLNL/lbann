# Dataset feature defeaults
# In general,  don't change these unless using cusom data - S.Z.

DATASET_CONFIG: dict = {
    "MAX_ATOMS": 100,  # The number of maximum atoms in CSD dataset
    "MAX_BONDS": 224,  # The number of maximum bonds in CSD dataset
    "ATOM_FEATURES": 133,
    "BOND_FEATURES": 147,
    "DATA_DIR": "/p/vast1/lbann/datasets/FLASK/CSD10K",
    "TARGET_FILE": "10k_dft_density_data.csv"  # Change to 10k_dft_hof_data.csv for heat of formation
}

# Hyperamaters used to set up trainer and MPN
# These can be changed freely 
HYPERPARAMETERS_CONFIG: dict = {
    "HIDDEN_SIZE": 300,
    "LR": 0.001,
    "BATCH_SIZE": 128,
    "EPOCH": 50,
    "MPN_DEPTH": 3
}
