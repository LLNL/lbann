# Experiments with node2vec

This work is focused on scaling node2vec on distributed systems, both
to achieve strong scaling and to handle very large graphs.

## Dependencies

- NVSHMEM: One-sided communication on Nvidia GPUs.

- HavoqGT: Distributed graph algorithms. https://github.com/KIwabuchi/havoqgt

- largescale_node2vec: Experimental implementation of distributed
  node2vec random walks. This is not yet publicly available.
  https://lc.llnl.gov/bitbucket/scm/havoq/largescale_node2vec.git

- (optional) SNAP: Baseline implementation of node2vec algorithm. Used
  for offline generation of random walks. https://github.com/snap-stanford/snap

Prior to building LBANN, download and install the dependencies:

```bash
# Paths
LBANN_DIR=/path/to/lbann
APP_DIR=${LBANN_DIR}/applications/graph/node2vec

# Download dependencies
cd ${LBANN_DIR}
git submodule update --init applications/graph/node2vec/havoqgt applications/graph/node2vec/largescale_node2vec applications/graph/node2vec/snap

# Build HavoqGT and largescale_node2vec
${APP_DIR}/build_havoqgt.sh

# Build SNAP (optional)
cd ${APP_DIR}/snap/examples/node2vec
make

```

When building LBANN itself, set the following CMake options:

- `CMAKE_CXX_STANDARD=17`
- `CMAKE_CUDA_STANDARD=17`
- `CXX_FLAGS="-isystem ${LBANN_DIR}/applications/graph/node2vec/largescale_node2vec/include -isystem ${LBANN_DIR}/applications/graph/node2vec/havoqgt/include"`
- `LBANN_WITH_NVSHMEM=1`
- `NVSHMEM_DIR=path/to/nvshmem`
- `LBANN_HAS_LARGESCALE_NODE2VEC=1`

## Running LBANN

`main.py` should be run from within a job allocation. If no options
are provided, it will download a graph from the SNAP website, process
it with HavoqGT, and perform node2vec with LBANN and the
largescale_node2vec random walker.
