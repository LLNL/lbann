import os.path

import utils.snap

graph_file = utils.snap.download_graph('ego-Facebook')
walk_file = os.path.join(os.path.dirname(graph_file), 'walk.txt')
graph = utils.snap.node2vec_walk(graph_file, walk_file,
                                 10, 4, 1.0, 1.0)
