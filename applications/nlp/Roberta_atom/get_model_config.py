import sys
import os
import warnings
import itertools
import time
import glob
import urllib.request
import argparse

import numpy as np
import torch

files = {
    "config.json": "https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/resolve/main/config.json",
    "pytorch_model.bin": "https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/resolve/main/pytorch_model.bin",
}
weights_dir = "pretrained_weights"


def download_file(url, fn):
    def report_hook(count, block_size, total_size):
        duration = int(time.time() - start_time)
        progress_size = int(count * block_size / (1024 ** 2))
        percent = min(int(count * block_size * 100 / total_size), 100)
        prog_bar = "|" + "#" * int(percent / 2) + "-" * (50 - int(percent / 2)) + "|"
        sys.stdout.write(
            f"\r{prog_bar} {percent}%, {progress_size} MB, {duration}s elapsed"
        )
        sys.stdout.flush()

    if os.path.exists(fn):
        warnings.warn(f"File '{fn}' already exists, skipping download")
    else:
        print(f"\n\nDownloading {fn} from {url}\n")
        start_time = time.time()
        urllib.request.urlretrieve(url, fn, report_hook)


def extract_weights(model, weights_dir):
    for name, weights in model.items():
        weights = np.array(weights).astype(np.float32)
        np.save(f"./{weights_dir}/{name}.npy", weights)


def process_weights(weights_dir):
    # Combine layernorm weights and bias to single file
    layernorm_files = glob.glob(f"./{weights_dir}/*LayerNorm*.npy")
    layernorm_groups = {}
    for fn in layernorm_files:
        base_fn = fn.split(".LayerNorm")[0]
        if base_fn in layernorm_groups:
            layernorm_groups[base_fn].append(fn)
        else:
            layernorm_groups[base_fn] = [fn]

    for base_fn, fns in layernorm_groups.items():
        weight_fn = [fn for fn in fns if "weight.npy" in fn][0]
        bias_fn = [fn for fn in fns if "bias.npy" in fn][0]

        weight_bias_vals = np.stack([np.load(weight_fn), np.load(bias_fn)]).T.copy()
        np.save(f"{base_fn}.layernorm.weightbias.npy", weight_bias_vals)

    # Combine layer_norm weights and bias to single file
    layer_norm_files = glob.glob(f"./{weights_dir}/*layer_norm*.npy")
    layer_norm_groups = {}
    for fn in layer_norm_files:
        base_fn = fn.split(".layer_norm")[0]
        if base_fn in layer_norm_groups:
            layer_norm_groups[base_fn].append(fn)
        else:
            layer_norm_groups[base_fn] = [fn]

    for base_fn, fns in layer_norm_groups.items():
        weight_fn = [fn for fn in fns if "weight.npy" in fn][0]
        bias_fn = [fn for fn in fns if "bias.npy" in fn][0]

        weight_bias_vals = np.stack([np.load(weight_fn), np.load(bias_fn)]).T.copy()
        np.save(f"{base_fn}.layer_norm.weightbias.npy", weight_bias_vals)

    # Transpose embedding layer weights
    embed_files = [
        glob.glob(f"{weights_dir}/{e}.npy")
        for e in (
            "*position_embeddings*",
            "*token_type_embeddings*",
            "*word_embeddings*",
        )
    ]
    embed_files = itertools.chain(*embed_files)
    for fn in embed_files:
        np.save(fn, np.load(fn).T.copy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-weights', action='store_true', help='avoids downloading model weights')
    args = parser.parse_args()

    if args.no_weights:
        del files['pytorch_model.bin']

    """Download model from huggingface"""
    for fn, url in files.items():
        download_file(url, fn)

    if not args.no_weights:
        """ Extract weights """
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        model = torch.load("pytorch_model.bin", map_location="cpu")
        extract_weights(model, weights_dir)

        """ Process weights for loading into LBANN """
        process_weights(weights_dir)
