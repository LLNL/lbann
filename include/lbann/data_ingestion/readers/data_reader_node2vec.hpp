////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READERS_NODE2VEC_HPP_INCLUDED
#define LBANN_DATA_READERS_NODE2VEC_HPP_INCLUDED

#include "lbann/data_ingestion/data_reader.hpp"
#ifdef LBANN_HAS_LARGESCALE_NODE2VEC

namespace lbann {

// Note (tym 4/8/20): Including largescale_node2vec in this header
// causes multiple definitions (I suspect it instantiates an object
// somewhere). However, node2vec_reader needs to store
// largescale_node2vec classes in unique_ptrs. To get around this, we
// implement derived classes in the source file and forward declare
// them in this header.
namespace node2vec_reader_impl {
class DistributedDatabase;
class EdgeWeightData;
class RandomWalker;
} // namespace node2vec_reader_impl

/** Adapter for HavoqGT distributed node2vec walker.
 *
 *  This is an experimental data reader intended for large-scale graph
 *  analytics with the node2vec algorithm. It requires building LBANN
 *  with HavoqGT and largescale_node2vec (not yet publicly available).
 *
 *  Data samples are lists of vertex indices: negative samples
 *  followed by a random walk. The negative samples are randomly
 *  chosen local vertices. The distribution of negative samples is
 *  periodically recomputed based on the number of times each vertex
 *  is visited.
 *
 *  @warning This is experimental.
 *
 */
class node2vec_reader : public generic_data_reader
{
public:
  node2vec_reader(std::string graph_file,
                  size_t epoch_size,
                  size_t walk_length,
                  double return_param,
                  double inout_param,
                  size_t num_negative_samples);
  node2vec_reader(const node2vec_reader&) = delete;
  node2vec_reader& operator=(const node2vec_reader&) = delete;
  ~node2vec_reader() override;
  node2vec_reader* copy() const override;

  std::string get_type() const override;

  const std::vector<int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;

  void load() override;

protected:
  bool fetch_data_block(CPUMat& X,
                        uint64_t block_offset,
                        uint64_t block_stride,
                        uint64_t mb_size,
                        El::Matrix<El::Int>& indices_fetched) override;
  bool fetch_label(CPUMat& Y, uint64_t data_id, uint64_t mb_idx) override;

private:
  /** Perform random walks, starting from random local vertices.
   *
   *  This uses IO RNG objects internally, so we have to make sure
   *  that the caller has acquired control of the IO RNG.
   */
  std::vector<std::vector<size_t>> run_walker(size_t num_walks,
                                              const locked_io_rng_ref&);

  /** Update noise distribution for negative sampling.
   *
   *  If a vertex has been visited @f$ \text{count} @f$ times, then
   *  its probability in the noise distribution is
   *  @f$ \text{count}^{0.75} @f$.
   */
  void update_noise_distribution();

  /** HavoqGT database for distributed graph. */
  std::unique_ptr<node2vec_reader_impl::DistributedDatabase>
    m_distributed_database;
  /** Edge weights for distributed graph. */
  std::unique_ptr<node2vec_reader_impl::EdgeWeightData> m_edge_weight_data;
  /** Manager for node2vec random walks on distributed graph. */
  std::unique_ptr<node2vec_reader_impl::RandomWalker> m_random_walker;

  /** Cache of random walks.
   *
   *  The random walker does not output the same number of walks as
   *  the number of start vertices. I presume this is because walks
   *  starting from local vertices may finish on remote processes and
   *  there is not a step to gather them to the original process. To
   *  handle cases where the walker returns no walks, we cache walks
   *  from previous mini-batch iterations.
   */
  std::deque<std::vector<size_t>> m_walks_cache;

  /** Global indices of local graph vertices. */
  std::vector<size_t> m_local_vertex_global_indices;
  /** Local indices of local graph vertices.
   *
   *  Inverse of @c m_local_vertex_global_indices.
   */
  std::unordered_map<size_t, size_t> m_local_vertex_local_indices;

  /** Number of times each local vertex has been visited in random
   *  walks.
   */
  std::vector<size_t> m_local_vertex_visit_counts;
  /** Noise distribution for negative sampling.
   *
   *  The values comprise a cumulative distribution function, i.e. the
   *  values are in [0,1], they are sorted in ascending order, and the
   *  last value is 1. To reduce communication, we only perform
   *  negative sampling with local vertices.
   *
   *  Computed in @c compute_noise_distribution.
   */
  std::vector<double> m_local_vertex_noise_distribution;

  /** Total number of times local vertices have been visited in random
   *  walks.
   *
   *  This is the sum of @c m_local_vertex_visit_counts.
   */
  size_t m_total_visit_count{0};
  /** The value of @c m_total_visit_count when the noise distribution
   *  was last computed.
   */
  size_t m_noise_visit_count{0};

  /** HavoqGT graph file.
   *
   *  This should be processed with HavoqGT's @c ingest_edge_list
   *  program and cached in shared memory.
   */
  std::string m_graph_file;

  /** @brief Number of data samples per "epoch".
   *
   *  LBANN assumes that datasets are static and have fixed size. Even
   *  though this data reader involves streaming data, it is
   *  convenient to group data samples into "epochs" so that we don't
   *  need to change the data model.
   */
  size_t m_epoch_size;
  /** @brief Length of each random walk. */
  size_t m_walk_length;
  /** @brief node2vec p parameter. */
  double m_return_param;
  /** @brief node2vec q parameter. */
  double m_inout_param;
  /** @brief Number of negative samples per data sample. */
  size_t m_num_negative_samples;
};

} // namespace lbann

#endif // LBANN_HAS_LARGESCALE_NODE2VEC
#endif // LBANN_DATA_READERS_NODE2VEC_HPP_INCLUDED
