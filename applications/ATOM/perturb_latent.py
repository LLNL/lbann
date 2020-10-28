#!/usr/bin/env python

"""
Script to generate perturbed latent vectors with added Gaussian noise
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse


def perturb_latent_vectors(latent_file, noise_factors):
    """
    Given a CSV file of latent vectors, generate a series of perturbed latent vector arrays
    by adding zero-mean Gaussian noise to the latent vector components, with SD
    equal to noise_factor standard deviations of the respective components. Output each array
    to a separate CSV file.
    """

    # Load the latent vector table, which includes an identifier or SMILES string in the first column
    #latent_df = pd.read_csv(latent_file)
    latent_df = pd.DataFrame(np.load(latent_file))
    print("Read %s" % latent_file)
    print("In File shape ", latent_df.shape)
    id_col = latent_df.columns.values[:102]
    latent_cols = latent_df.columns.values[102:]
    latent_dim = len(latent_cols)
    latent_rows = len(latent_df)
    latent_array = latent_df[latent_cols].values
    for noise_factor in noise_factors:
        latent_array = latent_df[latent_cols].values
        if noise_factor > 0.0:
            output_df = pd.DataFrame(latent_df[id_col].values)
            std_dev = [np.std(latent_array[:,i]) for i in range(latent_dim)]
            for i in range(latent_dim):
                latent_array[:,i] += np.random.randn(latent_rows) * std_dev[i] * noise_factor
                output_df[latent_cols[i]] = latent_array[:,i]
        else:
            output_df = latent_df
        output_file = '%s_noise_sd_%.2f.npy' % (os.path.splitext(latent_file)[0], noise_factor)
        print("Out df shape ", output_df.shape)
        np.save(output_file, output_df.to_numpy())
        print("Wrote %s" % output_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--latent_file", "-i", required=True)
    noise_factors = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0]

    args = parser.parse_args()
    perturb_latent_vectors(args.latent_file, noise_factors)


if __name__ == "__main__":
    main()
    sys.exit(0)
