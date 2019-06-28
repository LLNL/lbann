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
#ifdef LBANN_HAS_PYTHON
#include <string>
#include <Python.h>

namespace lbann {
namespace python {

/** @brief Singleton class to manage embedded Python session.
 *
 *  This mostly manages the initialization and finalization of the
 *  Python session. It is rarely necessary to interact with the
 *  singleton instance directly.
 *
 *  All static member functions are thread-safe.
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
 *  time. Make sure to acquire the GIL before calling Python C API
 *  functions. The GIL can be acquired recursively, i.e. you can
 *  acquire the GIL even if you already control it.
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

private:

  /** Python object pointer. */
  PyObject* m_ptr = nullptr;

};

} // namespace python
} // namespace lbann

#endif // LBANN_HAS_PYTHON
#endif // LBANN_UTILS_PYTHON_HPP_INCLUDED
