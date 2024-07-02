////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_DATA_READERS_PYTHON_DATASET_HPP_INCLUDED
#define LBANN_DATA_READERS_PYTHON_DATASET_HPP_INCLUDED

#include "lbann/data_ingestion/data_reader.hpp"
#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include "lbann/utils/python.hpp"

namespace lbann {

class python_dataset_reader : public generic_data_reader
{
public:
  python_dataset_reader(std::string dataset_path,
                        std::string module_dir,
                        uint64_t prefetch_factor,
                        bool shuffle)
    : generic_data_reader(shuffle),
      m_dataset_path(dataset_path),
      m_module_dir(module_dir),
      m_prefetch_factor(prefetch_factor)
  {}
  python_dataset_reader(const python_dataset_reader&) = default;
  python_dataset_reader& operator=(const python_dataset_reader&) = default;
  ~python_dataset_reader() override;
  python_dataset_reader* copy() const override
  {
    return new python_dataset_reader(*this);
  }

  std::string get_type() const override { return "python_dataset_reader"; }

  const std::vector<El::Int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_num_responses() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;
  int get_linearized_response_size() const override;

  void setup(int num_io_threads,
             observer_ptr<thread_pool> io_thread_pool) override;
  void load() override;
  void update(bool epoch_complete) override;

#ifdef LBANN_HAS_DISTCONV
  bool is_tensor_shuffle_required() const override
  {
    return m_tensor_shuffle_required;
  }
  void shuffle_responses(DataType* responses_ptr);
#endif // LBANN_HAS_DISTCONV

protected:
  bool fetch_data_block(std::map<data_field_type, CPUMat*>& input_buffers,
                        uint64_t current_position_in_data_set,
                        uint64_t block_offset,
                        uint64_t block_stride,
                        uint64_t sample_stride,
                        uint64_t mb_size,
                        El::Matrix<El::Int>& indices_fetched,
                        execution_mode mode = execution_mode::invalid) override;

private:
  void queue_epoch();
  void queue_samples(uint64_t samples_to_queue);

  /** @brief Path to the pickled dataset object. */
  std::string m_dataset_path;
  /** @brief Optional directory containing module with dataset definition. */
  std::string m_module_dir;
  /** @brief Number of samples to prefetch per worker. */
  int m_prefetch_factor;
  /** @brief Number of I/O threads. */
  int m_num_io_threads;
  /** @brief The current dataset shuffled minibatch offset. */
  uint64_t m_dataset_minibatch_offset;
  /** @brief The current dataset shuffled sample offset. */
  uint64_t m_dataset_sample_offset;
  /** @brief Number of samples requested this epoch. */
  uint64_t m_queued_samples;
  /** @brief Dimensions of data sample tensor. */
  std::vector<El::Int> m_sample_dims;
  /** @brief Size of label tensor. */
  El::Int m_num_labels;
  /** @brief Size of response tensor. */
  El::Int m_num_responses;
  /** @brief Number of data samples in data set. */
  El::Int m_num_samples;

  /** @brief User-provided Python dataset object.
   *
   *  The object must be a subclass of lbann.util.data.Dataset.
   */
  python::object m_dataset;

  /** @brief Python DataReader object.
   *
   *  Instance of lbann.util.data.DataReader used to launch worker processes.
   */
  python::object m_data_reader;

#ifdef LBANN_HAS_DISTCONV
  /** @brief Whether or not tensor needs shuffling for distconv. */
  bool m_tensor_shuffle_required = true;
  /** @brief The current number of minibatches in the epoch that have been
   * fetched and returned by fetch_data_block. */
  uint64_t m_fetched_minibatch_count;
#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_HAS_EMBEDDED_PYTHON
#endif // LBANN_DATA_READERS_PYTHON_DATASET_HPP_INCLUDED
