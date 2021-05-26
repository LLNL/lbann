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

namespace lbann {

communitygan_reader::communitygan_reader(
  std::string embedding_weights_name,
  std::string motif_file,
  std::string graph_file,
  size_t num_vertices,
  size_t motif_size,
  size_t walk_length,
  size_t epoch_size)
  : generic_data_reader(true),
    m_embedding_weights_name(std::move(embedding_weights_name)),
    m_motif_file(std::move(motif_file)),
    m_graph_file(std::move(graph_file)),
    m_num_vertices(num_vertices),
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
  if (m_cache_size <= 0) {
    m_cache_size = mb_size;
  }
  auto samples = generate_samples(io_rng);
  for (auto& sample : samples) {
    if (m_sample_cache.size() >= m_cache_size) {
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

  // Trainer info
  auto& comm = *get_comm();
  const size_t trainer_rank = comm.get_rank_in_trainer();
  const size_t trainer_size = comm.get_procs_per_trainer();

  m_motifs.clear();

  // Iterate through lines in file
  std::ifstream ifs(m_motif_file.c_str());
  std::string line;
  std::vector<size_t> vertices;
  vertices.reserve(m_motif_size);
  while (std::getline(ifs, line)) {

    // Objects for parsing line
    std::istringstream iss(line);
    size_t vertex;
    iss >> vertex; // First index is motif ID, so discard

    // Parse vertices in motif
    // Note: Keep track of vertices that are local or remote
    bool has_local_vertex = false;
    vertices.clear();
    for (size_t i=0; i<m_motif_size; ++i) {
      iss >> vertex;
      vertices.push_back(vertex);
      if (vertex % trainer_size == trainer_rank) {
        has_local_vertex = true;
      }
    }

    // Save motif if it has a local vertex
    if (has_local_vertex) {
      m_motifs.emplace_back(vertices);
    }

  }

  // Register with "setup CommunityGAN data reader" callback
  ::lbann::callback::setup_communitygan_data_reader::register_communitygan_data_reader(this);

  // Construct list of indices
  m_shuffled_indices.resize(m_epoch_size);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  select_subset_of_data();

}

std::vector<std::vector<size_t>> communitygan_reader::generate_samples(
  const locked_io_rng_ref&) {

  // Check that CommunityGAN walker has been initialized
  if (m_walker == nullptr) {
    LBANN_ERROR(
      "\"",get_type(),"\" data reader has uninitialized CommunityGAN walker. ",
      "Make sure model has \"setup CommunityGAN data reader\" callback.");
  }

  // Trainer info
  auto& comm = *get_comm();
  const size_t trainer_rank = comm.get_rank_in_trainer();
  const size_t trainer_size = comm.get_procs_per_trainer();

  // Randomly choose start vertices
  std::vector<int> starts;
  const size_t num_starts = m_cache_size / m_walks_per_vertex;
  starts.reserve(num_starts);
  size_t num_local_vertices = m_num_vertices / trainer_size;
  if (trainer_rank < m_num_vertices % trainer_size) {
    ++num_local_vertices;
  }
  for (size_t i=0; i<num_starts; ++i) {
    const size_t local_vertex_id = fast_rand_int(
      get_io_generator(),
      num_local_vertices);
    starts.push_back(local_vertex_id * trainer_size + trainer_rank);
  }

  // Perform random walks
  auto walks = m_walker->run(starts);

  // Construct data samples
  std::vector<std::vector<size_t>> samples;
  for (const auto& start_vertex_walks : walks) {
    const size_t start_vertex = start_vertex_walks.first;
    for (const auto& walk : start_vertex_walks.second) {

      // Remove duplicate vertices from walk
      std::unordered_set<size_t> walk_vertices(walk.cbegin(), walk.cend());
      if (walk_vertices.size() < m_motif_size) {
        continue;
      }
      walk_vertices.erase(start_vertex);

      // Construct sample with randomly chosen motif and walk
      /// @todo Choose motif that contains start vertex
      samples.emplace_back();
      auto& sample = samples.back();
      sample.reserve(m_motif_size+m_walk_length);
      const auto& motif = m_motifs[fast_rand_int(get_io_generator(), m_motifs.size())];
      sample.insert(sample.end(), motif.cbegin(), motif.cend());
      sample.push_back(start_vertex);
      sample.insert(sample.end(), walk_vertices.cbegin(), walk_vertices.cend());
      sample.resize(m_motif_size+m_walk_length, -1);

    }
  }

  // Shuffle samples
  std::shuffle(samples.begin(), samples.end(), get_io_generator());
  return samples;

}

} // namespace lbann

#endif // LBANN_HAS_COMMUNITYGAN_WALKER
