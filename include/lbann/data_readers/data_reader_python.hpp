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

#ifndef LBANN_DATA_READERS_PYTHON_HPP_INCLUDED
#define LBANN_DATA_READERS_PYTHON_HPP_INCLUDED

#include "data_reader.hpp"
#ifdef LBANN_HAS_PYTHON
#include "lbann/utils/python.hpp"

namespace lbann {

class python_reader : public generic_data_reader {
public:
  python_reader(std::string module,
                std::string module_dir,
                std::string sample_function,
                std::string num_samples_function,
                std::string sample_dims_function);
  python_reader(const python_reader&) = default;
  python_reader& operator=(const python_reader&) = default;
  ~python_reader() override;
  python_reader* copy() const override { return new python_reader(*this); }

  std::string get_type() const override {
    return "python_reader";
  }

  const std::vector<int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;

  void setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) override;
  void load() override;

protected:
  bool fetch_data_block(CPUMat& X,
                        El::Int thread_id,
                        El::Int mb_size,
                        El::Matrix<El::Int>& indices_fetched) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

private:

  /** @brief Dimensions of data sample tensor. */
  std::vector<El::Int> m_sample_dims;
  /** @brief Number of data samples in data set. */
  El::Int m_num_samples;

  /** @brief User-provided Python function to access data samples.
   *
   *  The function is expected to take one integer argument for the
   *  sample index. It must return an iterator that defines the
   *  entries in a data sample.
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

};

} // namespace lbann

#endif // LBANN_HAS_PYTHON
#endif // LBANN_DATA_READERS_PYTHON_HPP_INCLUDED
