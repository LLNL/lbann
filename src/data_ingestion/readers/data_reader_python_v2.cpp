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

#include "lbann/data_ingestion/readers/data_reader_python_v2.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/distconv.hpp"
#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include <algorithm>
#include <cstdio>
#include <regex>

#include <Python.h>

#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/python.hpp"

namespace lbann {

python_reader_v2::python_reader_v2(std::string dataset_path,
                                   bool shuffle)
  : generic_data_reader(shuffle)
{

  // Make sure Python is running and acquire GIL
  python::global_interpreter_lock gil;

  // Load the dataset object
  static El::Int instance_id = 0;
  instance_id++;
  const std::string dataset_name =
    ("_DATA_READER_PYTHON_CPP_dataset" +
     std::to_string(instance_id));
  std::string load_command = R"(
import pickle
with open('@dataset_path@', 'rb') as f:
  @dataset_name@ = pickle.load(f)
)";
  load_command = std::regex_replace(load_command,
                                    std::regex("\\@dataset_path\\@"),
                                    dataset_path);
  load_command = std::regex_replace(load_command,
                                    std::regex("\\@dataset_name\\@"),
                                    dataset_name);
  PyRun_SimpleString(load_command.c_str());
  python::object main_module = PyImport_ImportModule("__main__");
  m_dataset = PyObject_GetAttrString(main_module, dataset_name.c_str());
  python::check_error();

#ifdef LBANN_HAS_DISTCONV
  // Check if dataset supports distconv
  python::object lbann_data_module = PyImport_ImportModule("lbann.util.data");
  python::object distconv_dataset_class = PyObject_GetAttrString(lbann_data_module, "DistConvDataset");
  if (PyObject_IsInstance(m_dataset, distconv_dataset_class)) {
    m_tensor_shuffle_required = false;
    PyObject_SetAttrString(m_dataset, "rank", PyLong_FromLong(m_comm->get_rank_in_trainer()));
    PyObject_SetAttrString(m_dataset, "num_io_partitions", PyLong_FromLong(dc::get_number_of_io_partitions()));
  }
  python::check_error();
#endif // LBANN_HAS_DISTCONV

  // Get number of samples
  python::object num = PyObject_CallMethod(m_dataset, "__len__", nullptr);
  m_num_samples = PyLong_AsLong(num);
  python::check_error();

  // Get sample dimensions
  python::object sample_dims = PyObject_GetAttrString(m_dataset, "sample_dims");
  python::object dims;
  if (PyObject_HasAttrString(sample_dims, "sample")) {
    dims = PyObject_GetAttrString(sample_dims, "sample");
    dims = PyObject_GetIter(dims);
    for (auto d = PyIter_Next(dims); d != nullptr; d = PyIter_Next(dims)) {
      m_sample_dims.push_back(PyLong_AsLong(d));
      Py_DECREF(d);
    }
    python::check_error();
  }

  // Get label dimensions
  if (PyObject_HasAttrString(sample_dims, "label")) {
    dims = PyObject_GetAttrString(sample_dims, "label");
    m_num_labels = PyLong_AsLong(dims);
    python::check_error();
    generic_data_reader::set_has_labels(true);
  }

  // Get response dimensions
  if (PyObject_HasAttrString(sample_dims, "response")) {
    dims = PyObject_GetAttrString(sample_dims, "response");
    m_num_responses = PyLong_AsLong(dims);
    python::check_error();
    generic_data_reader::set_has_responses(true);
  }

  // Get sample access function
  m_sample_function = PyObject_GetAttrString(m_dataset, "__getitem__");
}

python_reader_v2::~python_reader_v2()
{
  if (python::is_active() && m_process_pool != nullptr) {
    python::global_interpreter_lock gil;
    PyObject_CallMethod(m_process_pool, "terminate", nullptr);
    PyObject_CallMethod(m_process_pool, "join", nullptr);
  }
}

const std::vector<El::Int> python_reader_v2::get_data_dims() const
{
  std::vector<El::Int> dims;
  for (const auto& d : m_sample_dims) {
    dims.push_back(d);
  }
  return dims;
}
int python_reader_v2::get_num_labels() const { return m_num_labels; }
int python_reader_v2::get_num_responses() const { return m_num_responses; }
int python_reader_v2::get_linearized_data_size() const
{
  return get_linear_size(get_data_dims());
}
int python_reader_v2::get_linearized_label_size() const
{
  return get_num_labels();
}
int python_reader_v2::get_linearized_response_size() const
{
  return get_num_responses();
}

bool python_reader_v2::fetch_data_block(
  std::map<data_field_type, CPUMat*>& input_buffers,
  uint64_t current_position_in_data_set,
  uint64_t block_offset,
  uint64_t block_stride,
  uint64_t sample_stride,
  uint64_t mb_size,
  El::Matrix<El::Int>& indices_fetched,
  const execution_mode mode)
{

  CPUMat& X = *(input_buffers[INPUT_DATA_TYPE_SAMPLES]);
  // Acquire Python GIL on first IO thread
  // Note: Do nothing on other IO threads.
  if (block_offset != 0) {
    return true;
  }
  python::global_interpreter_lock gil;

  // Check that shared memory array is large enough
  const uint64_t sample_size = get_linearized_data_size();
  const uint64_t array_size = PyObject_Length(m_shared_memory_array);
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
  for (uint64_t i = 0; i < mb_size; ++i) {
    El::Int sample_index =
      m_shuffled_indices[current_position_in_data_set + i * sample_stride];
    PyList_Append(
      args_list,
      python::object(Py_BuildValue("(l,l)", sample_index, i)));
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

  if (has_responses()) {
    CPUMat& Y = *(input_buffers[INPUT_DATA_TYPE_RESPONSES]);
    // Copy data from shared memory to output matrix
    CPUMat response_shared_memory_matrix(get_num_responses(),
                                mb_size,
                                m_response_shared_memory_array_ptr,
                                get_num_responses());
    El::Copy(response_shared_memory_matrix, Y);
  }

  return true;
}

void python_reader_v2::setup(int num_io_threads,
                             observer_ptr<thread_pool> io_thread_pool)
{
  generic_data_reader::setup(num_io_threads, io_thread_pool);

  // Acquire Python GIL
  python::global_interpreter_lock gil;

  // Import modules
  python::object main_module = PyImport_ImportModule("__main__");
  python::object ctypes_module = PyImport_ImportModule("ctypes");
  python::object multiprocessing_module =
    PyImport_ImportModule("multiprocessing");

  // Stop process pool if needed
  if (m_process_pool != nullptr) {
    PyObject_CallMethod(m_process_pool, "terminate", nullptr);
    m_process_pool = nullptr;
  }

  // Allocate shared memory array
  /// @todo Figure out more robust way to get max mini-batch size
  El::Int num_io_partitions = 1;
#ifdef LBANN_HAS_DISTCONV
  num_io_partitions = dc::get_number_of_io_partitions();
#endif // LBANN_HAS_DISTCONV
  const El::Int sample_size = get_linearized_data_size() / num_io_partitions;
  const El::Int mini_batch_size = get_trainer().get_max_mini_batch_size();
  std::string datatype_typecode;
  switch (sizeof(DataType)) {
  case 4:
    datatype_typecode = "f";
    break;
  case 8:
    datatype_typecode = "d";
    break;
  default:
    LBANN_ERROR("invalid data type for Python data reader "
                "(only float and double are supported)");
  }
  m_shared_memory_array = PyObject_CallMethod(multiprocessing_module,
                                              "RawArray",
                                              "(s, l)",
                                              datatype_typecode.c_str(),
                                              sample_size * mini_batch_size);

  // Get address of shared memory buffer
  python::object shared_memory_ptr =
    PyObject_CallMethod(ctypes_module,
                        "addressof",
                        "(O)",
                        m_shared_memory_array.get());
  m_shared_memory_array_ptr =
    reinterpret_cast<DataType*>(PyLong_AsLong(shared_memory_ptr));

  // Create global variables in Python
  // Note: The static counter makes sure variable names are unique.
  static El::Int instance_id = 0;
  instance_id++;
  const std::string sample_func_name =
    ("_DATA_READER_PYTHON_CPP_sample_function_wrapper" +
     std::to_string(instance_id));
  PyObject_SetAttrString(main_module,
                         sample_func_name.c_str(),
                         m_sample_function);
  python::check_error();
  const std::string shared_array_name =
    ("_DATA_READER_PYTHON_CPP_shared_memory_array" +
     std::to_string(instance_id));
  PyObject_SetAttrString(main_module,
                         shared_array_name.c_str(),
                         m_shared_memory_array);
  python::check_error();

  std::string response_shared_array_name = "None";
  if (has_responses()) {
    m_response_shared_memory_array = PyObject_CallMethod(multiprocessing_module,
                                                "RawArray",
                                                "(s, l)",
                                                datatype_typecode.c_str(),
                                                get_num_responses() * mini_batch_size);
    python::object response_shared_memory_ptr =
      PyObject_CallMethod(ctypes_module,
                          "addressof",
                          "(O)",
                          m_response_shared_memory_array.get());
    m_response_shared_memory_array_ptr = reinterpret_cast<DataType*>(PyLong_AsLong(response_shared_memory_ptr));
    response_shared_array_name =
      ("_DATA_READER_PYTHON_CPP_response_shared_memory_array" +
      std::to_string(instance_id));
    PyObject_SetAttrString(main_module,
                          response_shared_array_name.c_str(),
                          m_response_shared_memory_array);
    python::check_error();
  }

  // Create wrapper around sample function
  // Note: We attempt accessing the sample with the buffer protocol
  // since they can be copied more efficiently. If this fails, we just
  // iterate through the sample entries.
  /// @todo Handle multi-dimensional NumPy arrays.
  const std::string wrapper_func_name =
    ("_DATA_READER_PYTHON_CPP_sample_function" + std::to_string(instance_id));
  std::string wrapper_func_def = R"(
def @wrapper_func@(sample_index, array_offset):
    """Get data sample and copy to shared memory array."""

    # Get sample
    sample = @sample_func@(sample_index)

    input_buffer = memoryview(sample.sample)
    assert input_buffer.format == '@datatype_typecode@'
    input_buffer = input_buffer.cast('B').cast('@datatype_typecode@')
    output_buffer = memoryview(@shared_array@)
    output_buffer = output_buffer[array_offset*@sample_size@:(array_offset+1)*@sample_size@]
    output_buffer = output_buffer.cast('B').cast('@datatype_typecode@')
    output_buffer[:] = input_buffer

    # Get response
    if sample.response is not None:
        response = sample.response
        input_buffer = memoryview(response)
        output_buffer = memoryview(@response_shared_array@)
        output_buffer = output_buffer[array_offset*len(response):(array_offset+1)*len(response)]
        output_buffer = output_buffer.cast('B').cast('@datatype_typecode@')
        output_buffer[:] = input_buffer
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
                                        std::regex("\\@response_shared_array\\@"),
                                        response_shared_array_name);
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@sample_size\\@"),
                                        std::to_string(sample_size));
  wrapper_func_def = std::regex_replace(wrapper_func_def,
                                        std::regex("\\@datatype_typecode\\@"),
                                        datatype_typecode);
  PyRun_SimpleString(wrapper_func_def.c_str());
  python::check_error();
  m_sample_function_wrapper =
    PyObject_GetAttrString(main_module, wrapper_func_name.c_str());

  // Create initializer function for worker processes
  const std::string init_func_name = "_DATA_READER_PYTHON_CPP_init_function";
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
  python::check_error();
  python::object init_func =
    PyObject_GetAttrString(main_module, init_func_name.c_str());

  // Start Python process pool
  m_process_pool = PyObject_CallMethod(multiprocessing_module,
                                       "Pool",
                                       "(L,O)",
                                       num_io_threads,
                                       init_func.get());
}

void python_reader_v2::load()
{
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  select_subset_of_data();
}

} // namespace lbann

#endif // LBANN_HAS_EMBEDDED_PYTHON
