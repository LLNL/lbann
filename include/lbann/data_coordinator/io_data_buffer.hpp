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

#ifndef LBANN_IO_BUFFER_HPP_INCLUDED
#define LBANN_IO_BUFFER_HPP_INCLUDED

#include "lbann/data_readers/utils/input_data_type.hpp"

namespace lbann {

template <typename TensorDataType>
class data_buffer
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  /** Number of samples in the current mini-batch */
  int m_num_samples_fetched;
  /** Distributed matrix used to stage local data to layer output */
  std::map<data_field_type, std::unique_ptr<AbsDistMatrixType>> m_input_buffers;
  std::atomic<bool> m_fetch_data_in_background;
  std::future<void> m_data_fetch_future;
  /// 1-D Matrix of which indices were fetched in this mini-batch
  El::Matrix<El::Int> m_indices_fetched_per_mb;

  data_buffer(lbann_comm* comm)
    : m_num_samples_fetched(0), m_fetch_data_in_background(false)
  {
    m_input_buffers.clear();
  }

  data_buffer(const data_buffer& other)
    : m_num_samples_fetched(other.m_num_samples_fetched)
  {
    m_fetch_data_in_background.store(other.m_fetch_data_in_background);
    m_input_buffers.clear();
    // m_input_buffers.reserve(other.m_input_buffers.size());
    // for (const auto& ptr : other.m_input_buffers) {
    //   m_input_buffers.emplace_back(ptr ? ptr->Copy() : nullptr);
    // }
  }
  data_buffer& operator=(const data_buffer& other)
  {
    m_num_samples_fetched = other.m_num_samples_fetched;
    m_fetch_data_in_background.store(other.m_fetch_data_in_background);
    m_input_buffers.clear();
    // m_input_buffers.reserve(other.m_input_buffers.size());
    // for (const auto& ptr : other.m_input_buffers) {
    //   m_input_buffers.emplace_back(ptr ? ptr->Copy() : nullptr);
    // }
    return *this;
  }
  data_buffer* copy() const { return new data_buffer(*this); }

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** @brief Create a data parallel distributed matrix to hold the input data
   * for the field */
  void initialize_buffer_for_data_field(data_field_type const data_field,
                                        lbann_comm* comm);

  void set_background_fetching_in_progress(bool flag)
  {
    m_fetch_data_in_background = flag;
  }

  bool is_background_fetching_in_progress() const
  {
    return m_fetch_data_in_background;
  }

  /**
   * Return the sample indices fetched in the current mini-batch.
   */
  const El::Matrix<El::Int>* get_sample_indices_fetched_per_mb() const
  {
    return &m_indices_fetched_per_mb;
  }
  El::Matrix<El::Int>* get_sample_indices_fetched_per_mb()
  {
    return &m_indices_fetched_per_mb;
  }

  int num_samples_ready() const { return m_num_samples_fetched; }

  void set_data_fetch_future(std::future<void> future)
  {
    m_data_fetch_future = std::move(future);
  }

  std::future<void> get_data_fetch_future()
  {
    return std::move(m_data_fetch_future);
  }
};

} // namespace lbann

#endif // LBANN_IO_BUFFER_HPP_INCLUDED
