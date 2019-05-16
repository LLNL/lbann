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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_numpy_npz_conduit.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/data_readers/numpy_conduit_converter.hpp"
#include <unordered_set>
#include "lbann/utils/file_utils.hpp" // pad()
#include "lbann/utils/jag_utils.hpp"  // read_filelist(..) TODO should be move to file_utils
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"


namespace lbann {

numpy_npz_conduit_reader::numpy_npz_conduit_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

numpy_npz_conduit_reader::numpy_npz_conduit_reader(const numpy_npz_conduit_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

numpy_npz_conduit_reader& numpy_npz_conduit_reader::operator=(const numpy_npz_conduit_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void numpy_npz_conduit_reader::copy_members(const numpy_npz_conduit_reader &rhs) {
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);

  m_num_samples = rhs.m_num_samples;
  m_num_features = rhs.m_num_features;
  m_num_labels = rhs.m_num_labels;
  m_num_response_features = rhs.m_num_response_features;
  m_has_labels = rhs.m_has_labels;
  m_has_responses = rhs.m_has_responses;
  m_data_dims = rhs.m_data_dims;
  m_data_word_size = rhs.m_data_word_size;
  m_response_word_size = rhs.m_response_word_size;
  m_scaling_factor_int16 = rhs.m_scaling_factor_int16;
  m_filenames = rhs.m_filenames;
}

void numpy_npz_conduit_reader::load() {
  if(is_master()) {
    std::cout << "starting load" << std::endl;
  }

  options *opts = options::get();

  if (! (opts->get_bool("preload_data_store") || opts->get_bool("use_data_store"))) {
    LBANN_ERROR("numpy_npz_conduit_reader requires data_store; please pass either --use_data_store or --preload_data_store on the cmd line");
  }

  //dah - for now, I assume the input file contains, on each line, the complete
  //      pathname of an npz file. This will no doubt change in the future.
  //      I'd like to call load_list_of_samples(), but the sample_list class
  //      is too specialized -- it checks data in a manner particular to
  //      conduit, and that doesn't apply here.

  std::string infile = get_data_filename();
  read_filelist(m_comm, infile, m_filenames);

  // fills in: m_num_samples, m_num_features, m_num_response_features,
  // m_data_dims, m_data_word_size, m_response_word_size
  fill_in_metadata();

  if (m_num_labels == 0 && !opts->get_bool("preload_data_store") && opts->get_bool("use_data_store")) {
    LBANN_WARNING("when not preloading you must specify the number of labels in the prototext file if you are doing classification");
  }

  std::vector<int> local_list_sizes;
  if (opts->get_bool("preload_data_store")) {
    int np = m_comm->get_procs_per_trainer();
    int base_files_per_rank = m_filenames.size() / np;
    int extra = m_filenames.size() - (base_files_per_rank*np);
    if (extra > np) {
      LBANN_ERROR("extra > np");
    }
    local_list_sizes.resize(np, 0);
    for (int j=0; j<np; j++) {
      local_list_sizes[j] = base_files_per_rank;
      if (j < extra) {
        local_list_sizes[j] += 1;
      }
    }
  }

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  instantiate_data_store(local_list_sizes);

  // TODO: this may need fixing up for efficiency. If using an absolute
  //       num samples, or percentage of samples, and we've preloaded,
  //       this is wasteful and not what we want
  select_subset_of_data();
}

void numpy_npz_conduit_reader::preload_data_store() {
  double tm1 = get_time();
  m_data_store->set_preload();
  int rank = m_comm->get_rank_in_trainer();

  std::unordered_set<int> label_classes;
  for (size_t data_id=0; data_id<m_filenames.size(); data_id++) {
    if (m_data_store->get_index_owner(data_id) != rank) {
      continue;
    }

    conduit::Node node;
    numpy_conduit_converter::load_conduit_node(m_filenames[data_id], data_id, node);
    const char *char_ptr = node[LBANN_DATA_ID_STR(data_id) + "/frm/data"].value();
    const int* label_ptr = reinterpret_cast<const int*>(char_ptr);
    label_classes.insert(*label_ptr);
    m_data_store->set_conduit_node(data_id, node);
  }

  if (m_has_labels) {

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
      if(is_master()) {
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
  double tm2 = get_time();
  if (is_master()) {
    std::cout << "time to preload: " << tm2 - tm1 << " for role: " << get_role() << "\n";
  }
}

bool numpy_npz_conduit_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  Mat X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  } else {
    numpy_conduit_converter::load_conduit_node(m_filenames[data_id], data_id, node);
    //note: if testing, and test set is touched more than once, the following
    //      will through an exception TODO: relook later
    if (priming_data_store() || m_model->get_execution_mode() == execution_mode::testing) {
      m_data_store->set_conduit_node(data_id, node);
    }
  }

  const char *char_data = node[LBANN_DATA_ID_STR(data_id) + "/data/data"].value();
  char *char_data_2 = const_cast<char*>(char_data);

  if (m_data_word_size == 2) {
    // Convert int16 to DataType.
    short *data = reinterpret_cast<short*>(char_data_2);
    DataType *dest = X_v.Buffer();

    // OPTIMIZE
    LBANN_OMP_PARALLEL_FOR
      for(int j = 0; j < m_num_features; j++) {
        dest[j] = data[j] * m_scaling_factor_int16;
      }

  } else {
    void *data = (void*)char_data_2;
    std::memcpy(X_v.Buffer(), data, m_num_features * m_data_word_size);

    /*
    // the following is from data_reader_numpy_npz -- I don't think it's necessary
    if (m_data_word_size == 4) {
      float *f = reinterpret_cast<float*>(char_data_2);
      data = (void*)(f + data_id * m_num_features);
    } else if (m_data_word_size == 8) {
      double *d = reinterpret_cast<double*>(char_data_2);
      data = (void*)(d + data_id * m_num_features);
    }
    std::memcpy(X_v.Buffer(), data, m_num_features * m_data_word_size);
    */
  }

  return true;
}

bool numpy_npz_conduit_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  if (!m_has_labels) {
    LBANN_ERROR("numpy_npz_conduit_reader: do not have labels");
  }
  if (m_num_labels == 0) {
    LBANN_ERROR("num labels = 0. num_labels is only valid when run with --preload_data_store, *or* if your reader prototext contains a 'num_labels' field");
  }

  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  const char *char_data = node[LBANN_DATA_ID_STR(data_id)+ "/frm/data"].value();
  char *char_data_2 = const_cast<char*>(char_data);
  int *label = reinterpret_cast<int*>(char_data_2);
  Y(*label, mb_idx) = 1;

  return true;
}

bool numpy_npz_conduit_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  if (!m_has_responses) {
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
  } else {
    numpy_conduit_converter::load_conduit_node(m_filenames[data_id], data_id, node);
    if (priming_data_store()) {
      m_data_store->set_conduit_node(data_id, node);
    } else {
      LBANN_ERROR("you shouldn't be here; please contact Dave Hysom");
    }
  }

  const char *char_data = node[LBANN_DATA_ID_STR(data_id) + "/responses/data"].value();
  void *responses =  (void*)char_data;
  //char *char_data_2 = const_cast<char*>(char_data);
  //void *responses = (void*)
  /*
  if (m_response_word_size == 4) {
    responses = (void *) reinterpret_cast<float*>(char_data_2);
  } else if (m_response_word_size == 8) {
    responses = (void *) reinterpret_cast<double*>(char_data_2);
  } else {
    LBANN_ERROR("m_response_word_size= " + std::to_string(m_response_word_size) + "; should be 4 our 8");
  }
  */
  Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx + 1));
  std::memcpy(Y_v.Buffer(), responses,
              m_num_response_features * m_response_word_size);
  return true;
}

void numpy_npz_conduit_reader::fill_in_metadata() {
  int rank = m_comm->get_rank_in_trainer();
  // to avoid contention, each rank opens a separate file
  std::ifstream in(m_filenames[rank]);
  if (!in) {
    LBANN_ERROR("failed to open " + m_filenames[rank] + " for reading");
  }
  in.close();

  m_num_samples = m_filenames.size();
  if (is_master()) {
    std::cout << "num samples: " << m_num_samples << "\n";
  }

  int data_id = 0; //meaningless
  conduit::Node node;
  numpy_conduit_converter::load_conduit_node(m_filenames[rank], data_id, node);

  //fill in m_data_dims
  auto shape = node[LBANN_DATA_ID_STR(data_id) + "/data/shape"].as_uint64_array();
  int shape_num_elts = shape.number_of_elements();
  for (int k=1; k<shape_num_elts; k++) {
    m_data_dims.push_back(shape[k]);
  }
  m_num_features = std::accumulate(m_data_dims.begin() + 1,
                                   m_data_dims.end(),
                                   (unsigned) 1,
                                   std::multiplies<unsigned>());
  if (is_master()) {
    std::cout << "num features: " << m_num_features << "\n";
  }

  // Ensure we understand the word sizes
  size_t word_size = node[LBANN_DATA_ID_STR(data_id) + "/data/word_size"].value();
  if (!(word_size == 2 || word_size == 4 || word_size == 8)) {
    LBANN_ERROR("numpy_npz_conduit_reader: word size " +
                std::to_string(word_size) + " not supported");
  }
  m_data_word_size = word_size;
  if (is_master()) {
    std::cout << "data word size: " << m_data_word_size << "\n";
  }

  if (m_has_labels) {
    word_size = node[LBANN_DATA_ID_STR(data_id) + "/frm/word_size"].value();
    if (word_size != 4) {
      LBANN_ERROR("numpy_npz_conduit_reader: label should be in int32, but word_size= " + std::to_string(word_size));
    }
  }

  if (m_has_responses) {
    m_response_word_size = node[LBANN_DATA_ID_STR(data_id) + "/responses/word_size"].value();
    auto r_shape = node[LBANN_DATA_ID_STR(data_id) + "/responses/shape"].as_uint64_array();
    int n = r_shape.number_of_elements();
    m_num_response_features = 1;
    for (int k=1; k<n; k++) {
      m_num_response_features *= r_shape[k];
    }
    if (is_master()) {
      std::cout << "response word size: " << m_response_word_size << "\n";
      std::cout << "num response features: " << m_num_response_features<< "\n";
    }
  }
}

}  // namespace lbann
