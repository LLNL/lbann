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

#ifndef LBANN_DATA_READERS_COMMUNITYGAN_HPP_INCLUDED
#define LBANN_DATA_READERS_COMMUNITYGAN_HPP_INCLUDED

#include "data_reader.hpp"
#include "lbann/callbacks/setup_communitygan_data_reader.hpp"
#ifdef LBANN_HAS_COMMUNITYGAN_WALKER
#include "CommunityGANWalker_optimized.hpp"

namespace lbann {

class communitygan_reader : public generic_data_reader {
public:

  communitygan_reader(
    std::string embedding_weights_name,
    std::string motif_file,
    std::string graph_file,
    std::string start_vertices_file,
    size_t num_vertices,
    size_t motif_size,
    size_t walk_length,
    size_t walks_per_vertex,
    size_t path_limit,
    size_t epoch_size);
  communitygan_reader(const communitygan_reader&) = delete;
  communitygan_reader& operator=(const communitygan_reader&) = delete;
  ~communitygan_reader() override = default;
  communitygan_reader* copy() const override;

  std::string get_type() const override;

  const std::vector<int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;

  void load() override;

protected:
  bool fetch_data_block(
    std::map<data_field_type, CPUMat*>& input_buffers,
    El::Int block_offset,
    El::Int block_stride,
    El::Int mb_size,
    El::Matrix<El::Int>& indices_fetched) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

private:

  friend class ::lbann::callback::setup_communitygan_data_reader;

  /** @brief File with initial embedding values.
   *
   *  If provided, should be a space delimited text file. The first
   *  line specifies the matrix dimensions ((num vertices) x (embed
   *  dim)).
   */
  std::string m_embedding_weights_name;
  /** @brief Motif file.
   *
   *  Should be a space delimited text file. Each line corresponds to
   *  a motif: the first value is the motif ID (ignored) and the
   *  remaining are vertex IDs.
   */
  std::string m_motif_file;
  /** @brief Graph file. */
  std::string m_graph_file;
  /** @brief File containing start vertices for random walks.
   *
   *  If provided, should be a text file where each line is a vertex
   *  ID. All random walks will start from one of these vertices. If
   *  not provided, then random walks can start from any vertex.
   */
  std::string m_start_vertices_file;

  /** @brief Number of vertices in graph. */
  size_t m_num_vertices;
  /** @brief Number of vertices in motifs. */
  size_t m_motif_size;

  /** @brief Length of each random walk. */
  size_t m_walk_length;
  /** @brief Number of walks to start on each local vertex. */
  size_t m_walks_per_vertex;
  /** @brief Maximum number of vertices to visit when generating a walk */
  size_t m_path_limit;

  /** @brief Number of data samples per "epoch".
   *
   *  Random walks are generated once per epoch.
   */
  size_t m_epoch_size;

  std::unique_ptr<::CommunityGANWalker_optimized> m_walker;

  /** @brief Motifs that contain a local vertex. */
  std::vector<std::vector<size_t>> m_motifs;

  /** @brief Start vertices for random walks. */
  std::vector<size_t> m_start_vertices;

  /** @brief Data samples for current training epoch. */
  std::vector<std::vector<size_t>> m_cache;

  /** @brief Current position in sample cache. */
  size_t m_cache_pos = 0;

  /** @brief Number of local vertices in graph. */
  size_t get_num_local_vertices() const;

  /** @brief Perform random walks, starting from local vertices.
   *
   *  This uses IO RNG objects internally, so we have to make sure
   *  that the caller has acquired control of the IO RNG.
   */
  std::vector<std::vector<size_t>> generate_samples(
    const locked_io_rng_ref&);

};

} // namespace lbann

#endif // LBANN_HAS_COMMUNITYGAN_WALKER
#endif // LBANN_DATA_READERS_COMMUNITYGAN_HPP_INCLUDED
