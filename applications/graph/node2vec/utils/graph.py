import subprocess
import numpy as np

def max_vertex_index(graph_file):
    """Largest vertex index in graph.

    Vertices should be numbered consecutively from 0 to
    (num_vertices-1). LBANN will allocate unnecessary memory if there
    are any gaps in the indices. If any indices are negative, there
    may be mysterious errors.

    Args:
        graph_file (str): Uncompressed edge list file.

    Returns:
        int: Largest vertex index in graph.

    """
    command = [
        'awk',
        '{ i=i>$1?i:$1; i=i>$2?i:$2; } END { print(i); }',
        graph_file,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    return int(result.stdout.decode('utf-8'))
