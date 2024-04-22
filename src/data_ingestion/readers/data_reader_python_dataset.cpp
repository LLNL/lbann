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

#include "lbann/data_ingestion/readers/data_reader_python_dataset.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_ingestion/data_coordinator.hpp"
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

python_dataset_reader::~python_dataset_reader()
{
  if (python::is_active() && m_data_reader != nullptr) {
    python::global_interpreter_lock gil;
    python::object(PyObject_CallMethod(m_data_reader, "terminate", nullptr));
  }
}

const std::vector<El::Int> python_dataset_reader::get_data_dims() const
{
  std::vector<El::Int> dims;
  for (const auto& d : m_sample_dims) {
    dims.push_back(d);
  }
  return dims;
}
int python_dataset_reader::get_num_labels() const { return m_num_labels; }
int python_dataset_reader::get_num_responses() const { return m_num_responses; }
int python_dataset_reader::get_linearized_data_size() const
{
  return get_linear_size(get_data_dims());
}
int python_dataset_reader::get_linearized_label_size() const
{
  return get_num_labels();
}
int python_dataset_reader::get_linearized_response_size() const
{
  return get_num_responses();
}

bool python_dataset_reader::fetch_data_block(
  std::map<data_field_type, CPUMat*>& input_buffers,
  uint64_t current_position_in_data_set,
  uint64_t block_offset,
  uint64_t block_stride,
  uint64_t sample_stride,
  uint64_t mb_size,
  El::Matrix<El::Int>& indices_fetched,
  const execution_mode mode)
{
  // Note: Do nothing on other IO threads.
  if (block_offset != 0) {
    return true;
  }

  // Acquire Python GIL on first IO thread
  python::global_interpreter_lock gil;

  // Check that shared memory array is large enough
  uint64_t num_io_partitions = 1;
#ifdef LBANN_HAS_DISTCONV
  num_io_partitions = dc::get_number_of_io_partitions();
#endif // LBANN_HAS_DISTCONV
  const uint64_t sample_size = get_linearized_data_size() / num_io_partitions;

  El::Int sample_index;
  for (uint64_t i = 0; i < mb_size; ++i) {
    sample_index =
      m_shuffled_indices[current_position_in_data_set + i * sample_stride];
    indices_fetched.Set(i, 0, sample_index);
  }

  // Get the next batch from the Python data reader
  python::object batch =
    PyObject_CallMethod(m_data_reader, "get_batch", "(l)", mb_size);
  python::check_error();

  // Get samples
  // Note: we don't use python::objects here because PyDict_GetItemString
  // returns a borrowed reference (not a new pointer) and python::object
  // would try to decref it when it goes out of scope: SEGFAULT time
  auto sample_ptr = PyDict_GetItemString(batch, "sample_ptr");
  python::check_error();
  CPUMat shared_memory_matrix(
    sample_size,
    mb_size,
    static_cast<DataType*>(PyLong_AsVoidPtr(sample_ptr)),
    sample_size);
  CPUMat& X = *(input_buffers[INPUT_DATA_TYPE_SAMPLES]);
  El::Copy(shared_memory_matrix, X);

  // Get labels
  if (has_labels()) {
    auto label_ptr = PyDict_GetItemString(batch, "label_ptr");
    CPUMat label_shared_memory_matrix(
      get_num_labels(),
      mb_size,
      static_cast<DataType*>(PyLong_AsVoidPtr(label_ptr)),
      get_num_labels());
    CPUMat& Y = *(input_buffers[INPUT_DATA_TYPE_LABELS]);
    El::Copy(label_shared_memory_matrix, Y);
  }

  // Get responses
  if (has_responses()) {
    DataType* responses_ptr = static_cast<DataType*>(
      PyLong_AsVoidPtr(PyDict_GetItemString(batch, "response_ptr")));

#ifdef LBANN_HAS_DISTCONV
    if (!m_tensor_shuffle_required)
      shuffle_responses(responses_ptr);
#endif // LBANN_HAS_DISTCONV

    CPUMat response_shared_memory_matrix(get_num_responses(),
                                         mb_size,
                                         responses_ptr,
                                         get_num_responses());
    CPUMat& Y = *(input_buffers[INPUT_DATA_TYPE_RESPONSES]);
    El::Copy(response_shared_memory_matrix, Y);
  }

  // Prefetch the next minibatch asynchronously
  this->queue_samples(mb_size);

  return true;
}

#ifdef LBANN_HAS_DISTCONV
void python_dataset_reader::shuffle_responses(DataType* responses_ptr)
{
  // Shuffles the responses so that they are on the same ranks as the
  // non-distconv model predicitions to ensure correct loss calculations.
  // The shuffling calculations here assume that each sample in the distconv
  // layers and the shuffled non-distconv layers are indexed in order by all
  // samples in rank 0, then 1, and so on. The number of samples on each rank
  // (both in the distconv and non-distconv layers) is set such that they
  // are evenly distributed across all ranks and any additional samples
  // in a batch that can't be split evenly will be split evenly across the
  // first n ranks (or subsets of ranks in the distconv case).

  uint64_t rank = m_comm->get_rank_in_trainer();
  uint64_t nprocs = m_comm->get_procs_per_trainer();
  uint64_t trainer_rank = m_comm->get_trainer_rank();
  uint64_t num_io_partitions = dc::get_number_of_io_partitions();

  execution_mode mode = exec_mode_from_string(get_role());
  dataset& ds = get_trainer().get_data_coordinator().get_dataset(mode);
  uint64_t global_mb_size = ds.get_current_mini_batch_size();

  uint64_t local_mb_size = global_mb_size / nprocs;
  uint64_t extra_samples = global_mb_size % nprocs;
  uint64_t local_distconv_mb_size =
    global_mb_size / (nprocs / num_io_partitions);
  uint64_t distconv_extra_samples =
    global_mb_size % (nprocs / num_io_partitions);

  uint64_t send_rank, recv_rank, send_rank_count, recv_rank_count;
  send_rank = recv_rank = send_rank_count = recv_rank_count = 0;
  uint64_t send_rank_max_count =
    local_distconv_mb_size + (distconv_extra_samples > 0);
  uint64_t recv_rank_max_count = local_mb_size + (extra_samples > 0);
  for (uint64_t i = 0; i < global_mb_size; i++) {
    if (rank == send_rank) {
      if (send_rank == recv_rank) {
        std::memcpy(&responses_ptr[recv_rank_count * m_num_responses],
                    &responses_ptr[send_rank_count * m_num_responses],
                    m_num_responses * sizeof(DataType));
      }
      else {
#ifdef LBANN_BUILT_WITH_SPECTRUM
        // Due to a potential bug in Spectrum MPI's send, we must use ssend to
        // avoid hangs.
        EL_CHECK_MPI_CALL(
          MPI_Ssend(&responses_ptr[send_rank_count * m_num_responses],
                    m_num_responses * sizeof(DataType),
                    MPI_BYTE,
                    m_comm->get_world_rank(trainer_rank, recv_rank),
                    0,
                    m_comm->get_world_comm().GetMPIComm()));
#else
        m_comm->send(&responses_ptr[send_rank_count * m_num_responses],
                     m_num_responses,
                     trainer_rank,
                     recv_rank);
#endif
      }
    }
    else if (rank == recv_rank) {
      m_comm->recv(&responses_ptr[recv_rank_count * m_num_responses],
                   m_num_responses,
                   trainer_rank,
                   send_rank);
    }

    send_rank_count += 1;
    recv_rank_count += 1;
    if (send_rank_count == send_rank_max_count) {
      send_rank += num_io_partitions;
      send_rank_count = 0;
      if (send_rank / num_io_partitions == distconv_extra_samples)
        send_rank_max_count -= 1;
    }
    if (recv_rank_count == recv_rank_max_count) {
      recv_rank += 1;
      recv_rank_count = 0;
      if (recv_rank == extra_samples)
        recv_rank_max_count -= 1;
    }
  }
}
#endif // LBANN_HAS_DISTCONV

void python_dataset_reader::setup(int num_io_threads,
                                  observer_ptr<thread_pool> io_thread_pool)
{
  generic_data_reader::setup(num_io_threads, io_thread_pool);
  m_num_io_threads = num_io_threads;

  // Acquire Python GIL
  python::global_interpreter_lock gil;

  std::string datatype_typecode;
#if DataType == float
  datatype_typecode = "f";
#elif DataType == double
  datatype_typecode = "d";
#else
  LBANN_ERROR("invalid data type for Python data reader "
              "(only float and double are supported)");
#endif

  // Create Python data reader and worker processes
  python::object lbann_data = PyImport_ImportModule("lbann.util.data");
  m_data_reader = PyObject_CallMethod(lbann_data,
                                      "DataReader",
                                      "(O, l, l, s)",
                                      m_dataset.get(),
                                      num_io_threads,
                                      m_prefetch_factor,
                                      datatype_typecode.c_str());
  python::check_error();

  queue_epoch();
}

void python_dataset_reader::queue_samples(uint64_t samples_to_queue)
{
  // NOTE: ASSUMES GIL IS ALREADY TAKEN
  execution_mode mode = exec_mode_from_string(get_role());
  dataset& ds = get_trainer().get_data_coordinator().get_dataset(mode);

  // Get shuffled indices to be fetched by worker processes
  python::object inds_list = PyList_New(0);
  uint64_t num_samples = m_num_samples;
  uint64_t sample_stride = ds.get_sample_stride();
  uint64_t mini_batch_stride = ds.get_stride_to_next_mini_batch();
  uint64_t max_mb_size = ds.get_mini_batch_max();

  for (uint64_t i = 0; i < samples_to_queue; ++i) {
    uint64_t sample_ind = m_dataset_minibatch_offset * mini_batch_stride +
                          m_dataset_sample_offset * sample_stride;

    // We went over the entire epoch
    if (sample_ind >= num_samples)
      break;

    PyList_Append(
      inds_list,
      python::object(PyLong_FromLong(m_shuffled_indices[sample_ind])));

    m_dataset_sample_offset += 1;

    // Cycle minibatch offset
    if (m_dataset_sample_offset >= max_mb_size) {
      m_dataset_sample_offset = 0;
      m_dataset_minibatch_offset += 1;
    }
  }

  python::object(PyObject_CallMethod(m_data_reader,
                                     "queue_samples",
                                     "(O)",
                                     inds_list.get()));
  python::check_error();
}

void python_dataset_reader::queue_epoch()
{
  // Acquire Python GIL
  python::global_interpreter_lock gil;

  execution_mode mode = exec_mode_from_string(get_role());
  dataset& ds = get_trainer().get_data_coordinator().get_dataset(mode);

  // Resets the sample offset to the beginning of the epoch
  m_dataset_minibatch_offset = ds.get_base_offset();
  m_dataset_sample_offset = 0;

  queue_samples(m_prefetch_factor * m_num_io_threads);
}

void python_dataset_reader::load()
{
  // Make sure Python is running and acquire GIL
  python::global_interpreter_lock gil;

  if (!m_module_dir.empty()) {
    auto path = PySys_GetObject("path");
    PyList_Append(path, python::object(m_module_dir));
    python::check_error();
  }

  // Load Python dataset
  python::object pickle_module = PyImport_ImportModule("pickle");
  std::ifstream file(m_dataset_path, std::ios::binary);
  if (!file)
    LBANN_ERROR("failed to open dataset pickle file");
  std::string buffer(std::istreambuf_iterator<char>(file), {});
  python::object data =
    PyBytes_FromStringAndSize(buffer.c_str(), buffer.size());
  m_dataset = PyObject_CallMethod(pickle_module, "loads", "(O)", data.get());
  python::check_error();

#ifdef LBANN_HAS_DISTCONV
  // Check if dataset supports distconv
  python::object lbann_data_module = PyImport_ImportModule("lbann.util.data");
  python::object distconv_dataset_class =
    PyObject_GetAttrString(lbann_data_module, "DistConvDataset");
  if (PyObject_IsInstance(m_dataset, distconv_dataset_class)) {
    m_tensor_shuffle_required = false;
    PyObject_SetAttrString(m_dataset,
                           "rank",
                           PyLong_FromLong(get_comm()->get_rank_in_trainer()));
    PyObject_SetAttrString(m_dataset,
                           "num_io_partitions",
                           PyLong_FromLong(dc::get_number_of_io_partitions()));
  }
  python::check_error();
#endif // LBANN_HAS_DISTCONV

  // Get number of samples
  python::object num = PyObject_CallMethod(m_dataset, "__len__", nullptr);
  m_num_samples = PyLong_AsLong(num);
  python::check_error();

  // Get sample dimensions
  python::object sample_dims =
    PyObject_CallMethod(m_dataset, "get_sample_dims", nullptr);
  python::object dims;
  if (PyObject_HasAttrString(sample_dims, "sample")) {
    dims = PyObject_GetAttrString(sample_dims, "sample");
    dims = PyObject_GetIter(dims);
    for (python::object d = PyIter_Next(dims); d != nullptr;
         d = PyIter_Next(dims)) {
      m_sample_dims.push_back(PyLong_AsLong(d));
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

  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  select_subset_of_data();
}

void python_dataset_reader::update(bool epoch_complete)
{
  generic_data_reader::update(epoch_complete);

  if (epoch_complete)
    queue_epoch();
}

} // namespace lbann

#endif // LBANN_HAS_EMBEDDED_PYTHON
