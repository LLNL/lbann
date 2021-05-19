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
  const size_t trainer_rank = comm.get_trainer_rank();
  const size_t trainer_size = comm.get_procs_per_trainer();

  // Objects for parsing motif file
  m_motifs.clear();
  std::ifstream ifs(m_motif_file.c_str());
  std::istringstream iss;
  std::string line;
  std::vector<size_t> local_vertices, remote_vertices;
  local_vertices.reserve(m_motif_size);
  remote_vertices.reserve(m_motif_size);

  // Iterate through lines in file
  while (std::getline(ifs, line)) {

    // Objects for parsing line
    iss.str(line);
    size_t vertex;
    iss >> vertex; // First index is motif ID, so discard
    local_vertices.clear();
    remote_vertices.clear();

    // Parse vertices in motif
    // Note: Keep track of vertices that are local or remote
    for (size_t i=0; i<m_motif_size; ++i) {
      iss >> vertex;
      if (vertex % trainer_size == trainer_rank) {
        local_vertices.push_back(vertex);
      }
      else {
        remote_vertices.push_back(vertex);
      }
    }

    // Save motif if it has a local vertex
    if (!local_vertices.empty()) {
      m_motifs.emplace_back(local_vertices, remote_vertices);
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

  // Allocate memory for samples and walk starts
  std::vector<std::vector<size_t>> samples;
  std::vector<int> starts;
  samples.reserve(m_cache_size);
  starts.reserve(m_cache_size);

  // Randomly choose motifs and walk starts
  auto pick_random = [&] (const auto& vector) {
    return vector.at(fast_rand_int(get_io_generator(), vector.size()));
  };
  for (size_t i=0; i<m_cache_size; ++i) {

    // Allocate memory for sample
    samples.emplace_back();
    auto& sample = samples.back();
    sample.reserve(m_motif_size + m_walk_length);

    // Choose random motif
    const auto& motif = pick_random(m_motifs);
    const auto& local_vertices = motif.first;
    const auto& remote_vertices = motif.second;
    sample.insert(sample.end(), local_vertices.cbegin(), local_vertices.cend());
    sample.insert(sample.end(), remote_vertices.cbegin(), remote_vertices.cend());
    std::shuffle(sample.begin(), sample.end(), get_io_generator());

    // Choose random walk start
    starts.push_back(static_cast<int>(pick_random(local_vertices)));

  }

  // Perform random walks
  const auto walks = m_walker->run(starts);
  if (walks.size() != m_cache_size) {
    LBANN_ERROR(
      "CommunityGAN data reader expected ",
      m_cache_size," walks from walker, ",
      "but got ",walks.size());
  }
  for (size_t i=0; i<m_cache_size; ++i) {
    auto& sample = samples[i];
    /// @todo Cache all walks from walker
    for (const auto& vertex : pick_random(walks.at(starts[i]))) {
      sample.push_back(vertex);
    }
    sample.resize(m_motif_size + m_walk_length, -1);
  }
  return samples;

}

} // namespace lbann

#endif // LBANN_HAS_COMMUNITYGAN_WALKER
