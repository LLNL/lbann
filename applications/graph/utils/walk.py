import os
import os.path
import urllib.request
import gzip

import snap

class SnapGraph():

    def __init__(self, graph_file):
        self.graph = snap.LoadEdgeList(snap.PNGraph, graph_file)

    def walk(self):
        p = 1.0
        q = 1.0
        walk_length = 10

        nodes = snap.TIntIntPrPrV()
        embedding = snap.TIntFltVH()

        snap.node2vec(
            self.graph, p, q, -1, walk_length, 1, -1, -1, False, True,
            nodes, embedding
        )

    @staticmethod
    def download(name='ego-Facebook',
                 data_dir=None):
        """Download graph edgelist file from Stanford SNAP website.

        Args:
            name (str): Name of graph.
            data_dir (str, optional): Directory for downloading data.

        Returns:
            str: Uncompressed edge list file.

        """

        # Graphs from Stanford SNAP
        download_urls = {
            'ego-Facebook': 'http://snap.stanford.edu/data/facebook_combined.txt.gz',
        }

        # Make data directory if needed
        if not data_dir:
            root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            data_dir = os.path.join(root_dir, 'data', name)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        data_dir = os.path.realpath(data_dir)

        # Download and uncompress graph file
        gzip_file = os.path.join(data_dir, 'graph.txt.gz')
        txt_file = os.path.join(data_dir, 'graph.txt')
        urllib.request.urlretrieve(download_urls[name],
                                   filename=gzip_file)
        with gzip.open(gzip_file, 'rb') as in_file:
            with open(txt_file, 'wb') as out_file:
                out_file.write(in_file.read())

        return txt_file
