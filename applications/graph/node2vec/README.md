# Experiments with node2vec

This work is focused on scaling node2vec on distributed systems, both
to achieve strong scaling and to handle very large graphs.

## Dependencies

- SNAP: C++ package that includes baseline implementation of node2vec
  algorithm. Install with:

- HavoqGT: C++ framework that supports distributed graph algorithms.

- largescale_node2vec: Experimental implementation of distributed
  node2vec. This is not yet publicly available.

```bash
cd /path/to/lbann
git submodule update --init applications/graph/snap
cd applications/graph/snap
make
```
