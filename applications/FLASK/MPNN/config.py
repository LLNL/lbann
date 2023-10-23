# Dataset feature defeaults
# In general,  don't change these unless using cusom data - S.Z.

DATASET_CONFIG = {
    "MAX_ATOMS": 100,  # The number of maximum atoms in CSD dataset
    "MAX_BONDS": 224,  # The number of maximum bonds in CSD dataset
    "ATOM_FEATURES": 133,
    "BOND_FEATURES" : 147
}

# Hyperamaters used to set up trainer and MPN
# These can be changed freely 
HYPERPARAMETERS_CONFIG: dict = {
    "HIDDEN_SIZE":300,
    "LR": 0.001,
    "BATCH_SIZE" : 128,
    "EPOCH" : 50,
    "MPN_DEPTH": 3
}
