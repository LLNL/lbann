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

#include "lbann/data_readers/data_reader_python.hpp"
#ifdef LBANN_HAS_PYTHON
#include <cstdio>
#include <algorithm>
#include <regex>
#include "lbann/models/model.hpp"
#include "lbann/utils/python.hpp"

namespace lbann {

python_reader::python_reader(std::string module,
                             std::string module_dir,
                             std::string sample_function,
                             std::string num_samples_function,
                             std::string sample_dims_function)
  : generic_data_reader(true) {

  // Make sure Python is running and acquire GIL
  python::session::start_once();
  python::global_interpreter_lock gil;

  // Import Python module for data
  if (!module_dir.empty()) {
    auto path = PySys_GetObject("path");  // Borrowed reference
    PyList_Append(path, python::object(module_dir));
    python::session::check_error();
  }
  python::object data_module = PyImport_ImportModule(module.c_str());

  // Get number of samples
  python::object num_func
    = PyObject_GetAttrString(data_module, num_samples_function.c_str());
  python::object num = PyObject_CallObject(num_func, nullptr);
  m_num_samples = PyLong_AsLong(num);
  python::session::check_error();

  // Get sample dimensions
  python::object dims_func
    = PyObject_GetAttrString(data_module, sample_dims_function.c_str());
  python::object dims = PyObject_CallObject(dims_func, nullptr);
  dims = PyObject_GetIter(dims);
  for (auto d = PyIter_Next(dims); d != nullptr; d = PyIter_Next(dims)) {
    m_sample_dims.push_back(PyLong_AsLong(d));
    Py_DECREF(d);
  }
  python::session::check_error();

  // Get sample access function
  m_sample_function = PyObject_GetAttrString(data_module,
                                             sample_function.c_str());

}

python_reader::~python_reader() {
  if (python::session::is_active() && m_process_pool != nullptr) {
    python::global_interpreter_lock gil;
    PyObject_CallMethod(m_process_pool, "terminate", nullptr);
    PyObject_CallMethod(m_process_pool, "join", nullptr);
  }
}

const std::vector<int> python_reader::get_data_dims() const {
  std::vector<int> dims;
  for (const auto& d : m_sample_dims) {
    dims.push_back(d);
  }
  return dims;
}
int python_reader::get_num_labels() const {
  return 1;
}
int python_reader::get_linearized_data_size() const {
  const auto& dims = get_data_dims();
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<int>());
}
int python_reader::get_linearized_label_size() const {
  return get_num_labels();
}

bool python_reader::fetch_data_block(CPUMat& X,
                                     El::Int thread_id,
                                     El::Int mb_size,
                                     El::Matrix<El::Int>& indices_fetched) {

  // Acquire Python GIL on first IO thread
  // Note: Do nothing on other IO threads.
  if (thread_id != 0) { return true; }
  python::global_interpreter_lock gil;

  // Check that shared memory array is large enough
  const El::Int sample_size = get_linearized_data_size();
  const El::Int array_size = PyObject_Length(m_shared_memory_array);
  if (array_size < sample_size * mb_size) {
    std::stringstream err;
    err << "Python data reader attempted to load "
        << sample_size * mb_size * sizeof(DataType) << " B "
        << "into shared memory array, but only "
        << array_size * sizeof(DataType) << " B is available";
    LBANN_ERROR(err.str());
  }

  // Get arguments for sample access function
  python::object args_list = PyList_New(0);
  for (El::Int i = 0; i < mb_size; ++i) {
    El::Int sample_index = m_shuffled_indices[m_current_pos + i * m_sample_stride];
    El::Int array_offset = sample_size * i;
    PyList_Append(args_list,
                  python::object(Py_BuildValue("(l,l)",
                                               sample_index,
                                               array_offset)));
    indices_fetched.Set(i, 0, sample_index);
  }

  // Get samples using Python process pool
  python::object samples = PyObject_CallMethod(m_process_pool,
                                               "starmap",
                                               "(O,O)",
                                               m_sample_function_wrapper.get(),
                                               args_list.get());

  // Copy data from shared memory to output matrix
  CPUMat shared_memory_matrix(sample_size,
                              mb_size,
                              m_shared_memory_array_ptr,
                              sample_size);
  El::Copy(shared_memory_matrix, X);

  return true;
}

bool python_reader::fetch_label(CPUMat& Y, int data_id, int col) {
  return true;
}

void python_reader::setup(int num_io_threads,
                          std::shared_ptr<thread_pool> io_thread_pool) {
  generic_data_reader::setup(num_io_threads, io_thread_pool);

  // Acquire Python GIL
  python::global_interpreter_lock gil;

  // Import modules
  python::object main_module = PyImport_ImportModule("__main__");
  python::object ctypes_module = PyImport_ImportModule("ctypes");
  python::object multiprocessing_module
    = PyImport_ImportModule("multiprocessing");

  // Stop process pool if needed
  if (m_process_pool != nullptr) {
    PyObject_CallMethod(m_process_pool, "terminate", nullptr);
    m_process_pool = nullptr;
  }

  // Allocate shared memory array
  /// @todo Figure out more robust way to get max mini-batch size
  const El::Int sample_size = get_linearized_data_size();
  const El::Int mini_batch_size
    = generic_data_reader::get_model()->get_max_mini_batch_size();
  std::string datatype_typecode;
  switch (sizeof(DataType)) {
  case 4: datatype_typecode = "f"; break;
  case 8: datatype_typecode = "d"; break;
  default: LBANN_ERROR("invalid data type for Python data reader "
                       "(only float and double are supported)");
  }
  m_shared_memory_array
    = PyObject_CallMethod(multiprocessing_module,
                          "RawArray",
                          "(s, l)",
                          datatype_typecode.c_str(),
                          sample_size * mini_batch_size);

  // Get address of shared memory buffer
  python::object shared_memory_ptr
    = PyObject_CallMethod(ctypes_module,
                          "addressof",
                          "(O)",
                          m_shared_memory_array.get());
  m_shared_memory_array_ptr
    = reinterpret_cast<DataType*>(PyLong_AsLong(shared_memory_ptr));

  // Create global variables in Python
  // Note: The static counter makes sure variable names are unique.
  static El::Int instance_id = 0;
  instance_id++;
  const std::string sample_func_name
    = ("_DATA_READER_PYTHON_CPP_sample_function_wrapper"
       + std::to_string(instance_id));
  PyObject_SetAttrString(main_module,
                         sample_func_name.c_str(),
                         m_sample_function);
  python::session::check_error();
  const std::string shared_array_name
    = ("_DATA_READER_PYTHON_CPP_shared_memory_array"
       + std::to_string(instance_id));
  PyObject_SetAttrString(main_module,
                         shared_array_name.c_str(),
                         m_shared_memory_array);
  python::session::check_error();

  // Create wrapper around sample function
  // Note: We attempt accessing the sample with the buffer protocol
  // since they can be copied more efficiently. If this fails, we just
  // iterate through the sample entries.
  /// @todo Handle multi-dimensional NumPy arrays.
  const std::string wrapper_func_name
    = ("_DATA_READER_PYTHON_CPP_sample_function"
       + std::to_string(instance_id));
  std::string wrapper_func_def = R"(
def @wrapper_func@(sample_index, array_offset):
    """Get data sample and copy to shared memory array."""

    # Get sample
    sample = @sample_func@(sample_index)

    # Copy entries from sample to shared memory array
    # Note: We attempt to copy via the buffer protocol since it is
    # much more efficient than naively looping through the arrays.
    try:
        # Note: ctypes arrays explicitly specify their endianness, but
        # memoryview copies only work when the endianness is
        # explicitly set to the system default. We need to do some
        # type casting to get around this excessive error checking.
        input_buffer = memoryview(sample)
        output_buffer = memoryview(@shared_array@)
        output_buffer = output_buffer[array_offset:array_offset+@sample_size@]
        output_buffer = output_buffer.cast('B').cast('@datatype_typecode@')
        output_buffer[:] = input_buffer
    except:
        for i, val in enumerate(sample):
            @shared_array@[i + array_offset] = val
)";
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@wrapper_func\\@"),
                                        wrapper_func_name);
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@sample_func\\@"),
                                        sample_func_name);
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@shared_array\\@"),
                                        shared_array_name);
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@sample_size\\@"),
                                        std::to_string(sample_size));
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@datatype_typecode\\@"),
                                        datatype_typecode);
  PyRun_SimpleString(wrapper_func_def.c_str());
  python::session::check_error();
  m_sample_function_wrapper
    = PyObject_GetAttrString(main_module,
                             wrapper_func_name.c_str());

  // Create initializer function for worker processes
  const std::string init_func_name
    = "_DATA_READER_PYTHON_CPP_init_function";
  std::string init_func_def = R"(
def @init_func@():
    """Initialize worker process.

    Disables the LBANN signal handler since it reports a spurious error
    when the worker process recieves SIGTERM from the master process.

    """

    # Disable LBANN signal handler
    import signal
    for sig in range(signal.NSIG):
        try:
            signal.signal(sig, signal.SIG_DFL)
            pass
        except: pass
)";
  init_func_def = std::regex_replace(init_func_def,
                                     std::regex("\\@init_func\\@"),
                                     init_func_name);
  PyRun_SimpleString(init_func_def.c_str());
  python::session::check_error();
  python::object init_func
    = PyObject_GetAttrString(main_module,
                             init_func_name.c_str());

  // Start Python process pool
  m_process_pool = PyObject_CallMethod(multiprocessing_module,
                                       "Pool",
                                       "(L,O)",
                                       num_io_threads,
                                       init_func.get());

}

void python_reader::load() {
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

} // namespace lbann

#endif // LBANN_HAS_PYTHON
