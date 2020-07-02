"""Helper script to evaluate quality of node embeddings.

Converts the embedding weights computed by LBANN into a format that
can be read by Keita's evaluation script.

"""
import argparse
import os.path
import sys

import numpy as np

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    'embedding_file', type=str,
    help='node embeddings computed by LBANN', metavar='EMBEDDING_FILE')
parser.add_argument(
    'label_file', type=str,
    help='node labels', metavar='LABEL_FILE')
parser.add_argument(
    '--snap-embedding-file', default='results.emb', type=str,
    help='node embeddings in SNAP format', metavar='FILE')
args = parser.parse_args()

# Construct embedding file in SNAP's format
embeddings = np.loadtxt(args.embedding_file)
embeddings = np.transpose(embeddings)
with open(args.snap_embedding_file, 'w') as f:
    f.write(f'{embeddings.shape[0]} {embeddings.shape[1]}\n')
    for index, embedding in enumerate(embeddings):
        f.write(f'{index} {" ".join(str(x) for x in embedding)}\n')

# Evaluate embeddings with Keita's evaluation script
root_dir = os.path.dirname(os.path.realpath(__file__))
eval_script_dir = os.path.join(
    root_dir,
    'largescale_node2vec',
    'evaluation',
    'multi_label_classification'
)
sys.path.append(eval_script_dir)
import multi_label_classification
multi_label_classification.main([
    '-x', args.snap_embedding_file,
    '-y', args.label_file,
    '-r', 0.9,
    '-n', 10
])
