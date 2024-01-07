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

#ifndef LBANN_DATA_READERS_PYTHON_V2_HPP_INCLUDED
#define LBANN_DATA_READERS_PYTHON_V2_HPP_INCLUDED

#include "lbann/data_ingestion/data_reader.hpp"
#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include "lbann/utils/python.hpp"

namespace lbann {

class python_reader_v2 : public generic_data_reader
{
public:
  python_reader_v2(std::string dataset_path,
                   bool shuffle)
    : generic_data_reader(shuffle), m_dataset_path(dataset_path) {}
  python_reader_v2(const python_reader_v2&) = default;
  python_reader_v2& operator=(const python_reader_v2&) = default;
  ~python_reader_v2() override;
  python_reader_v2* copy() const override { return new python_reader_v2(*this); }

  std::string get_type() const override { return "python_reader_v2"; }

  const std::vector<El::Int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_num_responses() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;
  int get_linearized_response_size() const override;

  void setup(int num_io_threads,
             observer_ptr<thread_pool> io_thread_pool) override;
  void load() override;

#ifdef LBANN_HAS_DISTCONV
  bool is_tensor_shuffle_required() const override { return m_tensor_shuffle_required; }
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
  /** @brief Path to the pickled dataset object. */
  std::string m_dataset_path;
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
   *  The object must be a child of lbann.util.Dataset.
   */
  python::object m_dataset;

  /** @brief User-provided Python function to access data samples.
   *
   *  The function is expected to take one integer argument for the
   *  sample index. It must return a lbann.util.Sample object whose
   *  attributes can be converted to memoryviews.
   */
  python::object m_sample_function;

  /** @brief Wrapper function around sample access function.
   *
   *  This function will be executed on worker processes (see @c
   *  m_process_pool). It will obtain a data sample from @c
   *  m_sample_function and copy it into a @c m_shared_memory_array.
   */
  python::object m_sample_function_wrapper;

  /** @brief Pool of worker processes.
   *
   *  From the Python @c multiprocessing module.
   */
  python::object m_process_pool;

  /** @brief Shared memory array.
   *
   *  @c RawArray from the Python @c multiprocessing module.
   */
  python::object m_shared_memory_array;

  /** @brief Pointer into shared memory array.
   *
   *  Points to buffer for @c m_shared_memory_array.
   */
  DataType* m_shared_memory_array_ptr = nullptr;

  /** @brief Response shared memory array.
   *
   *  @c RawArray from the Python @c multiprocessing module.
   */
  python::object m_response_shared_memory_array;

  /** @brief Pointer into response shared memory array.
   *
   *  Points to buffer for @c m_shared_memory_array.
   */
  DataType* m_response_shared_memory_array_ptr = nullptr;

#ifdef LBANN_HAS_DISTCONV
  /** @brief Whether or not tensor needs shuffling for distconv. */
  bool m_tensor_shuffle_required = true;
#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_HAS_EMBEDDED_PYTHON
#endif // LBANN_DATA_READERS_PYTHON_V2_HPP_INCLUDED
