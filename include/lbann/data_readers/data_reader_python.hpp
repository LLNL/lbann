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
#include <Python.h>

namespace lbann {

namespace python {

/** @brief Singleton to manage embedded Python session.
 *
 *  This mostly manages the initialization and finalization of the
 *  Python session. It is rarely necessary to interact with the
 *  singleton instance directly.
 *
 *  All static functions are thread-safe.
 */
class session {
public:

  /** @brief Start embedded Python session if not already running.
   *  @details Does nothing if Python has already been started.
   */
  static void start_once();

  /** @brief Check if embedded Python session is running. */
  static bool is_active() noexcept;

  /** @brief Check if a Python error has occurred.
   *
   *  Throws an exception if a Python error is detected.
   *
   *  @param force_error Whether to force an exception to be thrown.
   */
  static void check_error(bool force_error = false);

  /** @brief Get singleton instance.
   *
   *  Initializes an embedded Python session the first time it is
   *  called.
   */
  static session& get();

  ~session();

private:

  /** @brief State on main Python thread. */
  PyThreadState* m_thread_state = nullptr;

  // Lifetime functions
  session();
  session(const session&) = delete;
  session& operator=(const session&) = delete;

};

/** @brief RAII wrapper for Python GIL.
 *
 *  The Python interpreter is not thread-safe, so it uses the "global
 *  interpreter lock" to ensure only one thread is executing at a
 *  time. Multithreading is achieved by periodically transferring
 *  control of the GIL between threads. This makes it hard to get
 *  meaningful speedups from simple multithreading. Certain
 *  operations, e.g. I/O and numerical kernels in NumPy, can be
 *  efficiently parallelized because they yield control of the GIL
 *  while working.
 */
class global_interpreter_lock {
public:
  global_interpreter_lock();
  ~global_interpreter_lock();
private:
  global_interpreter_lock(const global_interpreter_lock&) = delete;
  global_interpreter_lock& operator=(const global_interpreter_lock&) = delete;
  PyGILState_STATE m_gil_state;
};

/** @brief Wrapper around a Python object pointer.
 *
 *  Manages the reference count for a @c PyObject pointer and is
 *  implicitly convertible to the pointer. This is especially
 *  convenient for interacting with Python C API functions that @a
 *  borrow references to their arguments and return @a new references
 *  (this is the most common kind).
 *
 *  Handling reference counts is a tricky part of the Python C API. Be
 *  careful when using functions that @a steal references to their
 *  arguments or return @a borrowed references. See
 *
 *    https://docs.python.org/3.7/c-api/intro.html#reference-counts
 *
 *  for an explanation of reference counts.
 */
class object {
public:

  /** @brief Take ownership of a Python object pointer.
   *  @details @a Steals the reference.
   */
  object(PyObject* ptr);

  /** @brief Create a Python string. */
  object(const std::string& val);
  /** @brief Create a Python integer. */
  object(long val);
  /** @brief Create a Python floating point number. */
  object(double val);

  object() {}
  /** @details @a Borrows the reference. */
  object(const object& other);
  /** @details @a Borrows the reference. */
  object& operator=(const object& other);
  /** @details @a Steals the reference. */
  object(object&& other) noexcept;
  /** @details @a Steals the reference. */
  object& operator=(object&& other);
  ~object();

  /** @returns @a Borrowed reference. */
  inline PyObject* get() noexcept                  { return m_ptr; }
  /** @returns @a Borrowed reference. */
  inline const PyObject* get() const noexcept      { return m_ptr; }
  /** @returns @a Borrowed reference. */
  inline operator PyObject*() noexcept             { return get(); }
  /** @returns @a Borrowed reference. */
  inline operator const PyObject*() const noexcept { return get(); }

  /** @brief Release ownership of Python object pointer.
   *  @returns @a New reference.
   */
  PyObject* release() noexcept;

private:

  /** Python object pointer. */
  PyObject* m_ptr = nullptr;

};

} // namespace python

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
