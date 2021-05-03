////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/data_readers/data_reader_communitygan.hpp"
#ifdef LBANN_HAS_COMMUNITYGAN_WALKER
#include "lbann/utils/memory.hpp"
#include "lbann/utils/random.hpp"

#include "CommunityGANWalker.hpp"

namespace lbann {

communitygan_reader::communitygan_reader(
  std::string embedding_weights_name,
  std::string motif_file,
  std::string graph_file,
  size_t motif_size,
  size_t walk_length,
  size_t epoch_size)
  : generic_data_reader(true),
    m_embedding_weights_name(std::move(embedding_weights_name)),
    m_motif_file(std::move(motif_file)),
    m_graph_file(std::move(graph_file)),
    m_motif_size(motif_size),
    m_walk_length(walk_length),
    m_epoch_size(epoch_size) {
}

communitygan_reader* communitygan_reader::copy() const {
  LBANN_ERROR("can not copy communitygan_reader");
}

std::string communitygan_reader::get_type() const {
  return "communitygan_reader";
}

const std::vector<int> communitygan_reader::get_data_dims() const {
  std::vector<int> dims;
  dims.push_back(static_cast<int>(m_motif_size + m_walk_length));
  return dims;
}
int communitygan_reader::get_num_labels() const {
  return 1;
}
int communitygan_reader::get_linearized_data_size() const {
  const auto& dims = get_data_dims();
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<int>());
}
int communitygan_reader::get_linearized_label_size() const {
  return get_num_labels();
}

bool communitygan_reader::fetch_data_block(
  CPUMat& X,
  El::Int block_offset,
  El::Int block_stride,
  El::Int mb_size,
  El::Matrix<El::Int>& indices_fetched) {

  // Acquire IO RNG objects
  const auto io_rng = set_io_generators_local_index(block_offset);

  // Only run on first IO thread
  if (block_offset != 0) { return true; }
  const size_t mb_size_ = mb_size;

  // Generate samples and add to cache
  /// @todo Use larger cache and don't generate samples every
  /// mini-batch
  auto samples = generate_samples(mb_size, io_rng);
  const auto max_cache_size = std::max(mb_size_, m_sample_cache.size());
  for (auto& sample : samples) {
    if (m_sample_cache.size() >= max_cache_size) {
      m_sample_cache.pop_front();
    }
    m_sample_cache.emplace_back(std::move(sample));
  }

  // Populate output tensor
  /// @todo Parallelize
  for (size_t j=0; j<mb_size_; ++j) {
    const auto& sample = m_sample_cache[j % m_sample_cache.size()];
    for (size_t i=0; i<sample.size(); ++i) {
      X(i,j) = static_cast<float>(sample[i]);
    }
  }

  return true;
}

bool communitygan_reader::fetch_label(CPUMat& Y, int data_id, int col) {
  return true;
}

void communitygan_reader::load() {

#if 0 /// @todo Remove
  auto& comm = *get_comm();

  // Load graph data
  m_distributed_database = make_unique<DistributedDatabase>(
    ::havoqgt::db_open(),
    m_graph_file.c_str());
  auto& graph = *m_distributed_database->get_segment_manager()->find<Graph>("graph_obj").first;

  // Load edge data
  m_edge_weight_data.reset();
  auto* edge_weight_data = m_distributed_database->get_segment_manager()->find<EdgeWeightData::BaseType>("graph_edge_data_obj").first;
  if (edge_weight_data == nullptr) {
    m_edge_weight_data = make_unique<EdgeWeightData>(graph);
    m_edge_weight_data->reset(1.0);
    edge_weight_data = m_edge_weight_data.get();
  }
  comm.trainer_barrier();

  // Construct random walker
  constexpr bool small_edge_weight_variance = false;
  constexpr bool verbose = false;
  m_random_walker = make_unique<RandomWalker>(
    graph,
    *edge_weight_data,
    small_edge_weight_variance,
    m_walk_length,
    m_return_param,
    m_inout_param,
    comm.get_trainer_comm().GetMPIComm(),
    verbose);
  comm.trainer_barrier();

  // Get local vertices
  // Note: Estimate frequency of vertex visits using the vertex
  // degree, plus 1 for Laplace smoothing.
  const size_t num_local_vertices = graph.num_local_vertices();
  if (num_local_vertices == 0) {
    LBANN_ERROR("communitygan data reader loaded a graph with no local vertices");
  }
  m_local_vertex_global_indices.clear();
  m_local_vertex_global_indices.reserve(num_local_vertices);
  m_local_vertex_local_indices.clear();
  m_local_vertex_local_indices.reserve(num_local_vertices);
  m_local_vertex_visit_counts.clear();
  m_local_vertex_visit_counts.reserve(num_local_vertices);
  for (auto iter = graph.vertices_begin();
       iter != graph.vertices_end();
       ++iter) {
    const auto& vertex = *iter;
    const auto& degree = graph.degree(vertex);
    const auto& global_index = graph.locator_to_label(vertex);
    const auto& local_index = m_local_vertex_global_indices.size();
    m_local_vertex_global_indices.push_back(global_index);
    m_local_vertex_local_indices[global_index] = local_index;
    m_local_vertex_visit_counts.push_back(degree+1);
  }

  // Compute noise distribution for negative sampling
  update_noise_distribution();

  // Make sure walks cache has at least one walk
  const auto io_rng = set_io_generators_local_index(0);
  m_walks_cache.clear();
  do {
    auto walks = run_walker(1, io_rng);
    for (auto& walk : walks) {
      m_walks_cache.emplace_back(std::move(walk));
    }
  } while (comm.trainer_allreduce(m_walks_cache.empty() ? 1 : 0));


  // Construct list of indices
  m_shuffled_indices.resize(m_epoch_size);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  select_subset_of_data();
#endif // 0

}

std::vector<std::vector<size_t>> communitygan_reader::generate_samples(
  size_t num_samples,
  const locked_io_rng_ref&) {

#if 1 /// @todo Remove
  return std::vector<std::vector<size_t>>();
#else
  // HavoqGT graph
  const auto& graph = *m_distributed_database->get_segment_manager()->find<Graph>("graph_obj").first;
  const auto num_local_vertices = m_local_vertex_global_indices.size();

  // Randomly choose start vertices for random walks
  std::vector<Vertex> start_vertices;
  start_vertices.reserve(num_walks);
  for (size_t i=0; i<num_walks; ++i) {
    const auto& local_index = fast_rand_int(get_io_generator(),
                                            num_local_vertices);
    const auto& global_index = m_local_vertex_global_indices.at(local_index);
    start_vertices.push_back(graph.label_to_locator(global_index));
  }

  // Perform random walks
  const auto walks_vertices = m_random_walker->run_walker(start_vertices);

  // Convert walks to vertex indices
  std::vector<std::vector<size_t>> walks_indices;
  walks_indices.reserve(walks_vertices.size());
  for (const auto& walk_vertices : walks_vertices) {
    walks_indices.emplace_back();
    auto& walk_indices = walks_indices.back();
    walk_indices.reserve(walk_vertices.size());
    for (const auto& vertex : walk_vertices) {
      walk_indices.emplace_back(graph.locator_to_label(vertex));
    }
  }

  // Record visits to local vertices
  for (const auto& walk : walks_indices) {
    for (const auto& global_index : walk) {
      if (m_local_vertex_local_indices.count(global_index) != 0) {
        const auto& local_index = m_local_vertex_local_indices.at(global_index);
        ++m_local_vertex_visit_counts[local_index];
        ++m_total_visit_count;
      }
    }
  }

  return walks_indices;
#endif // 0

}

} // namespace lbann

#endif // LBANN_HAS_COMMUNITYGAN_WALKER
