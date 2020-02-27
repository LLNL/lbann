# Experiments with graph data

This work is focused on scaling graph embedding algorithms on
distributed systems, both to achieve strong scaling and to handle very
large graphs.

## Dependencies

- SNAP: C++ package that includes baseline implementation of node2vec
  algorithm. Install with:

```bash
cd /path/to/lbann
git submodule update --init applications/graph/snap
cd applications/graph/snap
make
```
