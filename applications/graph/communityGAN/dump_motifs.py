import os

def dump_motifs(prunejuice_output_dir, motif_file):
    """Convert PruneJuice output to a single motif file.

    First column is motif ID. Remaining columns are vertex IDs.

    """

    # Parse distributed motif files and find unique motifs
    # Note: Ignore first and last token in each line
    motifs = set()
    distributed_motif_dir = os.path.join(
        prunejuice_output_dir,
        'all_ranks_subgraphs',
    )
    for file_name in os.listdir(distributed_motif_dir):
        file_name = os.path.join(distributed_motif_dir, file_name)
        with open(file_name, 'r') as f:
            for line in f.readlines():
                tokens = line.split(',')
                motif = frozenset(int(t) for t in tokens[1:-1])
                motifs.add(motif)

    # Dump motifs to file
    with open(motif_file, 'w') as f:
        for i, motif in enumerate(motifs):
            f.write(f'{i} {" ".join(str(v) for v in motif)}\n')

if __name__ == '__main__':

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'prunejuice_output_dir', action='store', type=str,
        help='PruneJuice output'
    )
    parser.add_argument(
        'motif_file', action='store', type=str,
        help='output file'
    )
    args = parser.parse_args()

    # Convert PruneJuice output to a single motif file
    dump_motifs(args.prunejuice_output_dir, args.motif_file)
