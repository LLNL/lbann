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

#include "lbann/utils/python.hpp"
#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include "lbann/utils/exception.hpp"
#include <sstream>

#include <Python.h>

namespace lbann {
namespace python {

namespace {

/** @brief State on main Python thread after initialization. */
PyThreadState* init_thread_state = nullptr;

} // namespace

void initialize()
{

  // Thread-safe initialization with double-checked locking pattern
  if (!is_active()) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (!is_active()) {

      // Hack to display output from Python
      // Note: Python outputs didn't appear because MPI intercepts
      // stdout and stderr. See
      // https://stackoverflow.com/questions/29352485/python-print-not-working-when-embedded-into-mpi-program
      Py_UnbufferedStdioFlag = 1;

      // Initialize Python session and release GIL
      Py_Initialize();
#if !(PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 7)
      PyEval_InitThreads();
#endif
      init_thread_state = PyEval_SaveThread();
      if (!is_active()) {
        LBANN_ERROR("error initializing embedded Python session");
      }
    }
  }
}

void finalize()
{

  // Thread-safe finalization with double-checked locking pattern
  if (is_active()) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (is_active()) {

      // Take GIL and finalize Python session
      if (init_thread_state != nullptr) {
        PyEval_RestoreThread(init_thread_state);
        init_thread_state = nullptr;
      }
      Py_Finalize();

      // Check that Python session has been finalized
      if (is_active()) {
        LBANN_WARNING("error finalizing embedded Python session");
      }
    }
  }
}

bool is_active() { return Py_IsInitialized(); }

void check_error(bool force_error)
{
  global_interpreter_lock gil;
  if (force_error || PyErr_Occurred()) {

    // Get error information from Python session
    PyObject *type_ptr, *value_ptr, *traceback_ptr;
    PyErr_Fetch(&type_ptr, &value_ptr, &traceback_ptr);
    object type(type_ptr), value(value_ptr), traceback(traceback_ptr);

    // Construct error message
    std::ostringstream err;
    err << "detected Python error";
    if (value != nullptr) {
      object msg = PyObject_Repr(value);
      msg = PyUnicode_AsEncodedString(msg, "utf-8", "Error -");
      err << " (" << PyBytes_AS_STRING(msg.get()) << ")";
    }

    // Print Python traceback if available
    if (traceback != nullptr) {

      // Format traceback
      object module = PyImport_ImportModule("traceback");
      object func = PyObject_GetAttrString(module, "format_tb");
      object message =
        PyObject_CallMethod(module, "format_tb", "(O)", traceback.get());

      // Print traceback
      err << "\n\n"
          << "Python traceback:";
      object iter = PyObject_GetIter(message);
      for (object line = PyIter_Next(iter); line != nullptr;
           line = PyIter_Next(iter)) {
        const char* line_ = PyUnicode_AsUTF8(line);
        err << "\n" << (line_ ? line_ : "");
      }
    }

    // Clean up and throw exception
    PyErr_Restore(type.release(), value.release(), traceback.release());
    PyErr_Clear();
    LBANN_ERROR(err.str());
  }
}

// ---------------------------------------------
// global_interpreter_lock class
// ---------------------------------------------

global_interpreter_lock::global_interpreter_lock()
{
  initialize(); // Make sure Python is running
  m_gil_state = PyGILState_Ensure();
}

global_interpreter_lock::~global_interpreter_lock()
{
  if (is_active()) {
    PyGILState_Release(static_cast<PyGILState_STATE>(m_gil_state));
  }
}

// ---------------------------------------------
// object class
// ---------------------------------------------

object::object(PyObject* ptr) : m_ptr(ptr) { check_error(); }
object::object(const std::string& val)
{
  global_interpreter_lock gil;
  m_ptr = PyUnicode_FromStringAndSize(val.c_str(), val.size());
  check_error();
}
object::object(long val)
{
  global_interpreter_lock gil;
  m_ptr = PyLong_FromLong(val);
  check_error();
}
object::object(double val)
{
  global_interpreter_lock gil;
  m_ptr = PyFloat_FromDouble(val);
  check_error();
}

object::object(const object& other) : m_ptr(other.m_ptr)
{
  global_interpreter_lock gil;
  m_ptr = other.m_ptr;
  Py_XINCREF(m_ptr);
  check_error();
}

object& object::operator=(const object& other)
{
  global_interpreter_lock gil;
  Py_XDECREF(m_ptr);
  m_ptr = other.m_ptr;
  Py_XINCREF(m_ptr);
  check_error();
  return *this;
}

object::object(object&& other) noexcept : m_ptr(other.m_ptr)
{
  other.m_ptr = nullptr;
}

object& object::operator=(object&& other)
{
  global_interpreter_lock gil;
  Py_XDECREF(m_ptr);
  m_ptr = other.m_ptr;
  other.m_ptr = nullptr;
  check_error();
  return *this;
}

object::~object()
{
  if (is_active()) {
    global_interpreter_lock gil;
    Py_XDECREF(m_ptr);
  }
}

PyObject* object::release() noexcept
{
  auto old_ptr = m_ptr;
  m_ptr = nullptr;
  return old_ptr;
}

object::operator std::string()
{
  global_interpreter_lock gil;
  if (m_ptr == nullptr) {
    LBANN_ERROR("Attempted to convert Python object to std::string, "
                "but it has not been set");
  }
  object python_str = PyObject_Str(m_ptr);
  std::ostringstream ss;
  ss << PyUnicode_AsUTF8(python_str);
  check_error();
  return ss.str();
}

object::operator long()
{
  global_interpreter_lock gil;
  if (m_ptr == nullptr) {
    LBANN_ERROR("Attempted to convert Python object to long, "
                "but it has not been set");
  }
  auto val = PyLong_AsLong(m_ptr);
  check_error();
  return val;
}

object::operator double()
{
  global_interpreter_lock gil;
  if (m_ptr == nullptr) {
    LBANN_ERROR("Attempted to convert Python object to double, "
                "but it has not been set");
  }
  auto val = PyFloat_AsDouble(m_ptr);
  check_error();
  return val;
}

} // namespace python
} // namespace lbann

#endif // LBANN_HAS_EMBEDDED_PYTHON
