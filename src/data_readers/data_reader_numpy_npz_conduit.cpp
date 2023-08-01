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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_numpy_npz_conduit.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/file_utils.hpp" // pad()
#include "lbann/utils/jag_utils.hpp" // read_filelist(..) TODO should be move to file_utils
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/profiling.hpp"
#include "lbann/utils/timer.hpp"
#include <unordered_set>

namespace lbann {

numpy_npz_conduit_reader::numpy_npz_conduit_reader(const bool shuffle)
  : generic_data_reader(shuffle)
{}

numpy_npz_conduit_reader::numpy_npz_conduit_reader(
  const numpy_npz_conduit_reader& rhs)
  : generic_data_reader(rhs)
{
  copy_members(rhs);
}

numpy_npz_conduit_reader&
numpy_npz_conduit_reader::operator=(const numpy_npz_conduit_reader& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void numpy_npz_conduit_reader::copy_members(const numpy_npz_conduit_reader& rhs)
{
  if (rhs.m_data_store != nullptr) {
    m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);

  m_num_samples = rhs.m_num_samples;
  m_num_features = rhs.m_num_features;
  m_num_labels = rhs.m_num_labels;
  m_num_response_features = rhs.m_num_response_features;
  m_data_dims = rhs.m_data_dims;
  m_data_word_size = rhs.m_data_word_size;
  m_response_word_size = rhs.m_response_word_size;
  m_scaling_factor_int16 = rhs.m_scaling_factor_int16;
  m_filenames = rhs.m_filenames;
}

void numpy_npz_conduit_reader::load()
{
  if (get_comm()->am_world_master()) {
    std::cout << "starting load" << std::endl;
  }

  auto& arg_parser = global_argument_parser();

  if (!(arg_parser.get<bool>(LBANN_OPTION_PRELOAD_DATA_STORE) ||
        arg_parser.get<bool>(LBANN_OPTION_USE_DATA_STORE))) {
    LBANN_ERROR(
      "numpy_npz_conduit_reader requires data_store; please pass either "
      "--use_data_store or --preload_data_store on the cmd line");
  }

  // dah - for now, I assume the input file contains, on each line, the complete
  //       pathname of an npz file. This will no doubt change in the future.
  //       I'd like to call load_list_of_samples(), but the sample_list class
  //       is too specialized -- it checks data in a manner particular to
  //       conduit, and that doesn't apply here.

  std::string infile = get_data_filename();
  read_filelist(m_comm, infile, m_filenames);

  // fills in: m_num_features, m_num_response_features,
  // m_data_dims, m_data_word_size, m_response_word_size
  fill_in_metadata();

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  m_num_samples = m_shuffled_indices.size();

  if (m_num_labels == 0 &&
      !arg_parser.get<bool>(LBANN_OPTION_PRELOAD_DATA_STORE) &&
      arg_parser.get<bool>(LBANN_OPTION_USE_DATA_STORE)) {
    LBANN_WARNING("when not preloading you must specify the number of labels "
                  "in the prototext file if you are doing classification");
  }

  instantiate_data_store();

  select_subset_of_data();
}

void numpy_npz_conduit_reader::do_preload_data_store()
{
  double tm1 = get_time();

  if (get_comm()->am_world_master())
    std::cout
      << "Starting numpy_npz_conduit_reader::preload_data_store; num indices: "
      << m_shuffled_indices.size() << std::endl;

  size_t count = get_absolute_sample_count();
  double use_fraction = get_use_fraction();
  if (count != 0 || use_fraction != 1) {
    LBANN_ERROR("numpy_npz_conduit_reader currently assumes you are using 100% "
                "of the data set; you specified get_absolute_sample_count() = ",
                count,
                " and get_use_fraction() = ",
                use_fraction,
                "; please ask Dave Hysom to modify the code, if you want to "
                "use less than 100%");
  }

  int rank = m_comm->get_rank_in_trainer();

  std::unordered_set<int> label_classes;

  bool threaded =
    !global_argument_parser().get<bool>(LBANN_OPTION_DATA_STORE_NO_THREAD);

  // threaded mode
  if (threaded) {
    if (get_comm()->am_world_master()) {
      std::cout << "mode: data_store_thread\n";
    }
    std::shared_ptr<thread_pool> io_thread_pool =
      construct_io_thread_pool(m_comm, false);
    int num_threads = static_cast<int>(io_thread_pool->get_num_threads());

    // collect the set of indices that belong to this rank
    std::vector<std::unordered_set<int>> data_ids(num_threads);
    int j = 0;
    for (size_t data_id = 0; data_id < m_shuffled_indices.size(); data_id++) {
      int index = m_shuffled_indices[data_id];
      if (m_data_store->get_index_owner(index) != rank) {
        continue;
      }
      data_ids[j++].insert(index);
      if (j == num_threads) {
        j = 0;
      }
    }

    // load the samples
    for (int t = 0; t < num_threads; t++) {
      if (t == io_thread_pool->get_local_thread_id()) {
        continue;
      }
      else {
        io_thread_pool->submit_job_to_work_group(
          std::bind(&numpy_npz_conduit_reader::load_numpy_npz_from_file,
                    this,
                    data_ids[t],
                    label_classes));
      }
    }
    load_numpy_npz_from_file(data_ids[io_thread_pool->get_local_thread_id()],
                             label_classes);
    io_thread_pool->finish_work_group();
  } // end: threaded mode

  // non-threaded mode
  else {
    for (size_t data_id = 0; data_id < m_filenames.size(); data_id++) {
      if (m_data_store->get_index_owner(data_id) != rank) {
        continue;
      }

      conduit::Node node;
      load_npz(m_filenames[data_id], data_id, node);
      const char* char_ptr =
        node[LBANN_DATA_ID_STR(data_id) + "/frm/data"].value();
      const int* label_ptr = reinterpret_cast<const int*>(char_ptr);
      label_classes.insert(*label_ptr);
      m_data_store->set_conduit_node(data_id, node);
    }
  } // end: non-threaded mode

// Nikoli says we're not using labels, so I'm commenting this section out
// (this section is a mess, anyway)
#if 0
  if (m_supported_input_types[INPUT_DATA_TYPE_LABELS]) {

    // get max element. Yes, I know you can do this with, e.g, lambda
    // expressions and c++11 and etc, etc. But that's just B-ugly and
    // confusing
    int my_min = INT_MAX;
    int my_max = INT_MIN;
    for (auto t : label_classes) {
      if (t < my_min) { my_min = t; }
      if (t > my_max) { my_max = t; }
    }
    int trainer_min = m_comm->trainer_allreduce<int>(my_min, El::mpi::MIN);
    int trainer_max = m_comm->trainer_allreduce<int>(my_max, El::mpi::MAX);

    // dah - commenting out sanity checks, as I don't know if they're
    //       valid. Also, Nikoli says we're not using labels, so NA
    //       for now
#if 0
    // sanity checks
    if (trainer_min < 0) {
      LBANN_ERROR("trainer_min < 0");
    }
    if (trainer_max < 0) {
      LBANN_ERROR("trainer_max < 0");
    }
#endif

    // if we're using a subset of the data we may not have a contiguous
    // set of zero-based labels, so let's pretend like we do
    if (m_num_labels != 0) { //note: num_labels may be specified in the reader
      m_num_labels = trainer_max - trainer_min;
      if(get_comm()->am_world_master()) {
        std::cout << "num_labels: " << m_num_labels << "\n";
      }
    }

#if 0
    // Sanity checks.
    auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
    if (*minmax.first != 0) {
      LBANN_ERROR("numpy_reader: label classes are not indexed from 0");
    }
    if (*minmax.second != (int) label_classes.size() - 1) {
      LBANN_ERROR("numpy_reader: label classes are not contiguous");
    }
    m_num_labels = label_classes.size();
#endif
  }
#endif

  double tm2 = get_time();
  if (get_comm()->am_world_master()) {
    std::cout << "time to preload: " << tm2 - tm1 << " for role: " << get_role()
              << "\n";
  }
}

bool numpy_npz_conduit_reader::load_numpy_npz_from_file(
  const std::unordered_set<int>& data_ids,
  std::unordered_set<int>& label_classes)
{
  for (auto data_id : data_ids) {
    conduit::Node node;
    load_conduit_node(m_filenames[data_id], data_id, node);
    const char* char_ptr =
      node[LBANN_DATA_ID_STR(data_id) + "/frm/data"].value();
    const int* label_ptr = reinterpret_cast<const int*>(char_ptr);
    label_classes.insert(*label_ptr);
    m_data_store->set_conduit_node(data_id, node);
  }
  return true;
}

bool numpy_npz_conduit_reader::fetch_datum(Mat& X, int data_id, int mb_idx)
{
  LBANN_CALIPER_MARK_SCOPE("numpy_npz_conduit_reader::fetch_datum");
  Mat X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  }
  else {
    load_npz(m_filenames[data_id], data_id, node);
    // note: if testing, and test set is touched more than once, the following
    //       will through an exception TODO: relook later
    const auto& c = static_cast<const ExecutionContext&>(
      get_trainer().get_data_coordinator().get_execution_context());
    if (priming_data_store() ||
        c.get_execution_mode() == execution_mode::testing) {
      m_data_store->set_conduit_node(data_id, node);
    }
  }

  const char* char_data =
    node[LBANN_DATA_ID_STR(data_id) + "/data/data"].value();
  char* char_data_2 = const_cast<char*>(char_data);

  if (m_data_word_size == 2) {
    // Convert int16 to DataType.
    short* data = reinterpret_cast<short*>(char_data_2);
    DataType* dest = X_v.Buffer();

    // OPTIMIZE
    LBANN_OMP_PARALLEL_FOR
    for (int j = 0; j < m_num_features; j++) {
      dest[j] = data[j] * m_scaling_factor_int16;
    }
  }
  else {
    void* data = (void*)char_data_2;
    std::memcpy(X_v.Buffer(), data, m_num_features * m_data_word_size);

    /*
    // the following is from data_reader_numpy_npz -- I don't think it's
    necessary if (m_data_word_size == 4) { float *f =
    reinterpret_cast<float*>(char_data_2); data = (void*)(f + data_id *
    m_num_features); } else if (m_data_word_size == 8) { double *d =
    reinterpret_cast<double*>(char_data_2); data = (void*)(d + data_id *
    m_num_features);
    }
    std::memcpy(X_v.Buffer(), data, m_num_features * m_data_word_size);
    */
  }

  return true;
}

bool numpy_npz_conduit_reader::fetch_label(Mat& Y, int data_id, int mb_idx)
{
  if (!m_supported_input_types[INPUT_DATA_TYPE_LABELS]) {
    LBANN_ERROR("numpy_npz_conduit_reader: do not have labels");
  }
  if (m_num_labels == 0) {
    LBANN_ERROR("num labels = 0. num_labels is only valid when run with "
                "--preload_data_store, *or* if your reader prototext contains "
                "a 'num_labels' field");
  }

  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  const char* char_data =
    node[LBANN_DATA_ID_STR(data_id) + "/frm/data"].value();
  char* char_data_2 = const_cast<char*>(char_data);
  int* label = reinterpret_cast<int*>(char_data_2);
  Y(*label, mb_idx) = 1;

  return true;
}

bool numpy_npz_conduit_reader::fetch_response(Mat& Y, int data_id, int mb_idx)
{
  if (!m_supported_input_types[INPUT_DATA_TYPE_RESPONSES]) {
    LBANN_ERROR("numpy_npz_conduit_reader: do not have responses");
  }

  // assumes: fetch_datum for this data_id has previously been called,
  //          hence, the requested node will be in the data_store;
  //          this is for the case where we didn't preload. If we did
  //          preload, the requested nod should also be in the data_store
  //
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  }
  else {
    load_npz(m_filenames[data_id], data_id, node);
    if (priming_data_store()) {
      m_data_store->set_conduit_node(data_id, node);
    }
    else {
      LBANN_ERROR("you shouldn't be here; please contact Dave Hysom");
    }
  }

  const char* char_data =
    node[LBANN_DATA_ID_STR(data_id) + "/responses/data"].value();
  void* responses = (void*)char_data;
  // char *char_data_2 = const_cast<char*>(char_data);
  // void *responses = (void*)
  /*
  if (m_response_word_size == 4) {
    responses = (void *) reinterpret_cast<float*>(char_data_2);
  } else if (m_response_word_size == 8) {
    responses = (void *) reinterpret_cast<double*>(char_data_2);
  } else {
    LBANN_ERROR("m_response_word_size= " + std::to_string(m_response_word_size)
  + "; should be 4 our 8");
  }
  */
  Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx + 1));
  std::memcpy(Y_v.Buffer(),
              responses,
              m_num_response_features * m_response_word_size);
  return true;
}

void numpy_npz_conduit_reader::fill_in_metadata()
{
  int rank = m_comm->get_rank_in_trainer();
  // to avoid contention, each rank opens a separate file
  size_t my_file = rank;
  if (my_file >= m_filenames.size()) {
    my_file = 0;
  }
  std::ifstream in(m_filenames[my_file]);
  if (!in) {
    LBANN_ERROR("failed to open " + m_filenames[my_file] + " for reading");
  }
  in.close();
  m_num_samples = m_filenames.size();
  if (get_comm()->am_world_master()) {
    std::cout << "num samples: " << m_num_samples << "\n";
  }

  int data_id = 0; // meaningless
  conduit::Node node;
  load_npz(m_filenames[my_file], data_id, node);

  // fill in m_data_dims
  auto shape =
    node[LBANN_DATA_ID_STR(data_id) + "/data/shape"].as_uint64_array();
  int shape_num_elts = shape.number_of_elements();
  for (int k = 1; k < shape_num_elts; k++) {
    m_data_dims.push_back(shape[k]);
  }
  m_num_features = std::accumulate(m_data_dims.begin(),
                                   m_data_dims.end(),
                                   (unsigned)1,
                                   std::multiplies<unsigned>());
  if (get_comm()->am_world_master()) {
    std::cout << "num features: " << m_num_features << "\n";
  }

  // Ensure we understand the word sizes
  size_t word_size =
    node[LBANN_DATA_ID_STR(data_id) + "/data/word_size"].value();
  if (!(word_size == 2 || word_size == 4 || word_size == 8)) {
    LBANN_ERROR("numpy_npz_conduit_reader: word size " +
                std::to_string(word_size) + " not supported");
  }
  m_data_word_size = word_size;
  if (get_comm()->am_world_master()) {
    std::cout << "data word size: " << m_data_word_size << "\n";
  }

  if (m_supported_input_types[INPUT_DATA_TYPE_LABELS]) {
    word_size = node[LBANN_DATA_ID_STR(data_id) + "/frm/word_size"].value();
    if (word_size != 4) {
      LBANN_ERROR(
        "numpy_npz_conduit_reader: label should be in int32, but word_size= " +
        std::to_string(word_size));
    }
  }

  if (m_supported_input_types[INPUT_DATA_TYPE_RESPONSES]) {
    m_response_word_size =
      node[LBANN_DATA_ID_STR(data_id) + "/responses/word_size"].value();
    auto r_shape =
      node[LBANN_DATA_ID_STR(data_id) + "/responses/shape"].as_uint64_array();
    int n = r_shape.number_of_elements();
    m_num_response_features = 1;
    for (int k = 1; k < n; k++) {
      m_num_response_features *= r_shape[k];
    }
    if (get_comm()->am_world_master()) {
      std::cout << "response word size: " << m_response_word_size << "\n";
      std::cout << "num response features: " << m_num_response_features << "\n";
    }
  }
}

void numpy_npz_conduit_reader::load_conduit_node(const std::string filename,
                                                 int data_id,
                                                 conduit::Node& output,
                                                 bool reset)
{

  try {
    if (reset) {
      output.reset();
    }

    std::vector<size_t> shape;
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filename);

    for (auto&& t : a) {
      cnpy::NpyArray& b = t.second;
      if (b.shape[0] != 1) {
        LBANN_ERROR("lbann currently only supports one sample per npz file; "
                    "this file appears to contain " +
                      std::to_string(b.shape[0]) + " samples; (",
                    filename);
      }
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/word_size"] =
        b.word_size;
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/fortran_order"] =
        b.fortran_order;
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/num_vals"] =
        b.num_vals;
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/shape"] = b.shape;

      if (b.data_holder->size() / b.word_size != b.num_vals) {
        LBANN_ERROR("b.data_holder->size() / b.word_size (" +
                    std::to_string(b.data_holder->size()) + " / " +
                    std::to_string(b.word_size) + ") != b.num_vals (" +
                    std::to_string(b.num_vals));
      }

      // conduit makes a copy of the data, hence owns the data, hence it
      // will be properly deleted when then conduit::Node is deleted
      char* data = b.data_holder->data();
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/data"].set_char_ptr(
        data,
        b.word_size * b.num_vals);
    }
  }
  catch (...) {
    // note: npz_load throws std::runtime_error, but I don't want to assume
    //       that won't change in the future
    LBANN_ERROR("failed to open " + filename + " during cnpy::npz_load");
  }
}

void numpy_npz_conduit_reader::load_npz(const std::string filename,
                                        int data_id,
                                        conduit::Node& output)
{

  try {
    output.reset();

    std::vector<size_t> shape;
    m_npz_cache[data_id] = cnpy::npz_load(filename);
    std::map<std::string, cnpy::NpyArray>& a = m_npz_cache[data_id];

    for (auto&& t : a) {
      cnpy::NpyArray& b = t.second;
      if (b.shape[0] != 1) {
        LBANN_ERROR("lbann currently only supports one sample per npz file; "
                    "this file appears to contain " +
                      std::to_string(b.shape[0]) + " samples; (",
                    filename);
      }
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/word_size"] =
        b.word_size;
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/fortran_order"] =
        b.fortran_order;
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/num_vals"] =
        b.num_vals;
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/shape"] = b.shape;

      if (b.data_holder->size() / b.word_size != b.num_vals) {
        LBANN_ERROR("b.data_holder->size() / b.word_size (" +
                    std::to_string(b.data_holder->size()) + " / " +
                    std::to_string(b.word_size) + ") != b.num_vals (" +
                    std::to_string(b.num_vals));
      }

      conduit::uint8* data =
        reinterpret_cast<conduit::uint8*>(b.data_holder->data());
      output[LBANN_DATA_ID_STR(data_id) + "/" + t.first + "/data"]
        .set_external_uint8_ptr(data, b.word_size * b.num_vals);
    }
  }
  catch (...) {
    // note: npz_load throws std::runtime_error, but I don't want to assume
    //       that won't change in the future
    LBANN_ERROR("failed to open " + filename + " during cnpy::npz_load");
  }
}

} // namespace lbann
