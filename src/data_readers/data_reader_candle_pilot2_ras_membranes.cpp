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

#include "lbann/data_readers/data_reader_candle_pilot2_ras_membranes.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_node.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include <unordered_set>
#include "lbann/utils/file_utils.hpp" // pad()
#include "lbann/utils/jag_utils.hpp"  // read_filelist(..) TODO should be move to file_utils
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/lbann_library.hpp"

namespace lbann {

candle_pilot2_ras_membranes_reader::candle_pilot2_ras_membranes_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

candle_pilot2_ras_membranes_reader::candle_pilot2_ras_membranes_reader(const candle_pilot2_ras_membranes_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

candle_pilot2_ras_membranes_reader& candle_pilot2_ras_membranes_reader::operator=(const candle_pilot2_ras_membranes_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void candle_pilot2_ras_membranes_reader::copy_members(const candle_pilot2_ras_membranes_reader &rhs) {
  if (is_master()) {
    std::cout << "Starting candle_pilot2_ras_membranes_reader::copy_members\n";
  }
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);
  m_filenames = rhs.m_filenames;
  m_data_id_map = rhs.m_data_id_map;
  m_datum_shapes = rhs.m_datum_shapes;
  m_datum_num_words = rhs.m_datum_num_words;
  m_num_features = rhs.m_num_features;
  m_num_labels = rhs.m_num_labels;
  m_num_response_features = rhs.m_num_response_features;
  m_data_dims = rhs.m_data_dims;

  m_sample_list.copy(rhs.m_sample_list);
  //m_list_per_trainer = rhs.m_list_per_trainer;
  //m_list_per_model = rhs.m_list_per_model;
}

void candle_pilot2_ras_membranes_reader::load() {
  options *opts = options::get();

  if (! opts->get_bool("preload_data_store")) {
    LBANN_ERROR("candle_pilot2_ras_membranes_reader requires data_store; please pass --preload_data_store on the cmd line");
  }

  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  // TODO: need to fix code in src/proto/proto_common.cpp 
  if ((m_leading_reader != this) && (m_leading_reader != nullptr)) {
    // The following member variables of the leadering reader should have been
    // copied when this was copy-constructed: m_sample_list, and m_open_hdf5_files
    return;
  }

  m_shuffled_indices.clear();

  //TODO: "Load the sample list" and "Merge all of the sample lists"
  //      is cut-n-paste from data_reader_jag_conduit.cpp; this should
  //      be refactored to avoid duplication

  // Load the sample list
  double tm1 = get_time();
  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string sample_list_file = data_dir + get_data_index_list();
  size_t stride = m_comm->get_procs_per_trainer();
  size_t offset = m_comm->get_rank_in_trainer();
  m_sample_list.load(sample_list_file, stride, offset);
  double tm2 = get_time();
  if (is_master()) {
    std::cout << "Time to load sample list: " << tm2 - tm1 << std::endl;
  }

  // Merge all of the sample lists
  tm1 = get_time();
  m_sample_list.all_gather_packed_lists(*m_comm);
  if (opts->has_string("write_sample_list") && m_comm->am_trainer_master()) {
    {
      const std::string msg = " writing sample list " + sample_list_file;
      LBANN_WARNING(msg);
    }
    std::stringstream s;
    std::string basename = get_basename_without_ext(sample_list_file);
    std::string ext = get_ext_name(sample_list_file);
    s << basename << "." << ext;
    m_sample_list.write(s.str());
  }
  if (is_master()) {
    std::cout << "time for all_gather_packed_lists: " << get_time() - tm1 << std::endl;
  }

  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  fill_in_metadata();
  read_normalization_data();

  instantiate_data_store();
  select_subset_of_data();
}

void candle_pilot2_ras_membranes_reader::do_preload_data_store() {
  if (is_master()) std::cout << "Starting candle_pilot2_ras_membranes_reader::do_preload_data_store; num indices: " << utils::commify(m_shuffled_indices.size()) << " role: " << get_role() << std::endl;

#if 0
==========================================================================
data types, from python+numpy:

  rots          numpy.float64  RAS rotation angle    
  states        numpy.float64  the assigned state
  tilts         numpy.float64  RAS tilt angle
  density_sig1  numpy.float64  13x13x14 lipid density data
  frames        numpy.int64    list of frame ids
  bbs           numpy.float32  184x3 xyz coordinates for 
                               each of the 184 RAS backbone beads
  probs         numpy.float64  probability to be in each of the three states

  Notes: 
    1. Adam talks about 'frames,' which is equivalent, in lbannese, to 'sample'
    2. the "frames" field is simply a set of sequential integers, 0 .. 539

==========================================================================
#endif

  bool verbose = options::get()->get_bool("verbose");
  size_t nn = 0;
  int np = m_comm->get_procs_per_trainer();
  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    try {
      conduit::Node & node = m_data_store->get_empty_node(index);
      load_conduit_node(index, node, true);
      normalize(node);
      m_data_store->set_preloaded_conduit_node(index, node);
      nn++;
      if (verbose && is_master() && nn % 10 == 0) {
        std::cout << "approx. " << utils::commify(nn*np) << " samples loaded\n";
      }
    } catch (conduit::Error const& e) {
      LBANN_ERROR("Error:  trying to load the node ", index, " and got ", e.what());
    }
  }
  /// Once all of the data has been preloaded, close all of the file handles
  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    m_sample_list.close_if_done_samples_file_handle(index);
  }
}

bool candle_pilot2_ras_membranes_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  } else {
    m_sample_list.open_samples_file_handle(data_id);
    load_conduit_node(data_id, node);
    if (priming_data_store()) {
      m_data_store->set_conduit_node(data_id, node);
    }
  }

  const double *data = node[LBANN_DATA_ID_STR(data_id) + "/density_sig1/data"].value();
  size_t n = m_datum_num_words["density_sig1"];
  for (size_t j = 0; j < n; ++j) {
    X(j, mb_idx) = data[j];
  }  

#if 0
Notes from Adam:
The keras model that I gave you only looks at the density_sig1 data as input data and it uses the states data as labels.  We¿ll want to also extract bbs to merge that with density_sig1 in various ways as input data in future models that we¿re putting together.

 The probs field can be useful as an alternate label if building a regression model instead of a classification model.  I¿ve also been using the probs field as a filter on the training data to only consider those input data whose state probability exceeds some threshold.

  So that works out to:

   bb, density_sig1 - datum
   states           - label
   probs            - used as a filter to include/exclude certain samples

#endif
  return true;
}

bool candle_pilot2_ras_membranes_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  double label = node[LBANN_DATA_ID_STR(data_id) + "/states/data"].value();
  Y.Set(label, mb_idx, 1);
  return true;
}

bool candle_pilot2_ras_membranes_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("candle_pilot2_ras_membranes_reader: do not have responses");
  return true;
}

void candle_pilot2_ras_membranes_reader::fill_in_metadata() {
  int index = 0;
  const sample_t& s = m_sample_list[index];
  sample_file_id_t id = s.first;
  m_sample_list.open_samples_file_handle(index, true);
  auto h = m_sample_list.get_samples_file_handle(id);
  conduit::Node node;
  const std::string path = "/";
  read_node(h, path, node);

  // Fill in m_datum_shapes and m_datum_num_words
  conduit::NodeConstIterator itr = node.children();
  while (itr.has_next()) {
    const conduit::Node& n_child = itr.next();
    conduit::NodeConstIterator itr_2 = n_child.children();
    while (itr_2.has_next()) {
      const conduit::Node& n_child_2 = itr_2.next();
      const std::string &name = n_child_2.name();
      size_t nelts = n_child_2["shape"].dtype().number_of_elements();
      const size_t *shape_ptr = n_child_2["shape"].as_unsigned_long_ptr();
      for (size_t i=0; i<nelts; i++) {
        m_datum_shapes[name].push_back(shape_ptr[i]);
      }
      m_datum_num_words[name] = n_child_2["size"].value();
    }
    break;
  }

  // Count m_num_features
  m_num_features = 1;
  for (auto t : m_datum_shapes["density_sig1"]) {
    m_data_dims.push_back(t);
    m_num_features *= t;
  }
}


#ifdef _USE_IO_HANDLE_
bool candle_pilot2_ras_membranes_reader::has_path(const data_reader_jag_conduit::file_handle_t& h,
                                       const std::string& path) const {
  return m_sample_list.is_file_handle_valid(h) && h->has_path(path);
}

void candle_pilot2_ras_membranes_reader::read_node(const data_reader_jag_conduit::file_handle_t& h,
                                        const std::string& path,
                                        conduit::Node& n) const {
  if (!h) {
    return;
  }
  h->read(path, n);
}
#else
bool candle_pilot2_ras_membranes_reader::has_path(const hid_t& h, const std::string& path) const {
  return (m_sample_list.is_file_handle_valid(h) &&
          conduit::relay::io::hdf5_has_path(h, path));
}

void candle_pilot2_ras_membranes_reader::read_node(const hid_t& h, const std::string& path, conduit::Node& n) const {
  conduit::relay::io::hdf5_read(h, path, n);
}
#endif //#ifdef _USE_IO_HANDLE_

void candle_pilot2_ras_membranes_reader::read_normalization_data() {
  options *opts = options::get();
  if (! opts->has_string("normalization")) {
    m_normalize = normalization_type::none;
    if (is_master()) {
      std::cout << "NOT Normalizing data" << std::endl;
    }
    return;
  }
  if (opts->get_bool("min_max")) {
    m_normalize = normalization_type::min_max;
    if (is_master()) {
      std::cout << "Normalizing data using min-max" << std::endl;
    }
  } else {
    m_normalize = normalization_type::std_dev;
    if (is_master()) {
      std::cout << "Normalizing data using z-score" << std::endl;
    }
  }

  const std::string fn = options::get()->get_string("normalization");
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", fn, " for reading");
  }
  std::string line;
  getline(in, line); //discard header
  m_max_min.reserve(14);
  m_min.reserve(14);
  m_mean.reserve(14);
  m_std_dev.reserve(14);
  double v_max, v_min, v_mean, v_std_dev;
  while (in >> v_max >> v_min >> v_mean >> v_std_dev) { 
    m_min.push_back(v_min);
    m_max_min.push_back(v_max - v_min);
    m_mean.push_back(v_mean);
    m_std_dev.push_back(v_std_dev);
  }
  in.close();
  if (m_min.size() != 14) {
    LBANN_ERROR("normalization.size() = ", m_min.size(), "; should be 14");
  }
}

void candle_pilot2_ras_membranes_reader::load_conduit_node(const size_t data_id, conduit::Node &node, bool pre_open_fd) {
  const sample_t& s = m_sample_list[data_id];
  const std::string& sample_name = s.second;
  sample_file_id_t id = s.first;
  m_sample_list.open_samples_file_handle(data_id, pre_open_fd);
  auto h = m_sample_list.get_samples_file_handle(id);
  read_node(h, sample_name, node['/' + LBANN_DATA_ID_STR(data_id)]);
}

void candle_pilot2_ras_membranes_reader::normalize(conduit::Node &node) {
  if (m_normalize == normalization_type::none) {
    return;
  }
  //TODO
}

}  // namespace lbann
