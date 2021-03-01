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

#ifndef LBANN_UTILS_PYTHON_HPP_INCLUDED
#define LBANN_UTILS_PYTHON_HPP_INCLUDED

#include "lbann/base.hpp"
#ifdef LBANN_HAS_EMBEDDED_PYTHON

#include <Python.h>

#include <mutex>
#include <string>

namespace lbann {
namespace python {

/** @brief Start embedded Python session.
 *
 *  Does nothing if Python has already been started. This function is
 *  thread-safe.
 *
 *  Be warned that restarting Python after it has been ended is a bad
 *  idea since any Python objects left over from the first session
 *  will be invalid in the second. Expect segfaults.
 */
void initialize();

/** @brief End embedded Python session.
 *
 *  Does nothing if Python is not running. This function is
 *  thread-safe.
 */
void finalize();

/** @brief Check if embedded Python session is running. */
bool is_active();

/** @brief Check if a Python error has occurred.
 *
 *  If a Python error is detected, then the Python error indicator is
 *  cleared and a C++ exception is thrown. The GIL is acquired
 *  internally.
 *
 *  @param force_error Whether to force an exception to be thrown.
 */
void check_error(bool force_error = false);

/** @brief RAII wrapper for Python GIL.
 *
 *  The Python interpreter is not thread-safe, so it uses the "global
 *  interpreter lock" to ensure only one thread is executing at a
 *  time. Make sure to acquire the GIL before calling Python C API
 *  functions. The GIL can be acquired recursively, i.e. you can
 *  acquire the GIL even if you already control it.
 *
 *  If an Python session is not running, one is started.
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
 *  borrow references and return @a new references (this is the most
 *  common kind).
 *
 *  This class is @a not thread-safe. However, it's best practice to
 *  acquire the GIL before doing any Python operations, so access will
 *  typically be serialized.
 *
 *  Handling reference counts is a tricky part of the Python C API. Be
 *  especially careful with functions that @a steal references or
 *  return @a borrowed references. See
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

  /** Convert Python @c str to C++ @c std::string. */
  operator std::string();
  /** Convert Python @c int to C++ @c long. */
  operator long();
  /** Convert Python @c float to C++ @c double. */
  operator double();

private:

  /** Python object pointer. */
  PyObject* m_ptr = nullptr;

};

} // namespace python
} // namespace lbann

#endif // LBANN_HAS_EMBEDDED_PYTHON
#endif // LBANN_UTILS_PYTHON_HPP_INCLUDED
