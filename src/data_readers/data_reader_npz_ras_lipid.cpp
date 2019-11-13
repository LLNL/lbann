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

#include "lbann/data_readers/data_reader_npz_ras_lipid.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include <unordered_set>
#include "lbann/utils/file_utils.hpp" // pad()
#include "lbann/utils/jag_utils.hpp"  // read_filelist(..) TODO should be move to file_utils
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/lbann_library.hpp"

namespace lbann {

ras_lipid_conduit_data_reader::ras_lipid_conduit_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

ras_lipid_conduit_data_reader::ras_lipid_conduit_data_reader(const ras_lipid_conduit_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

ras_lipid_conduit_data_reader& ras_lipid_conduit_data_reader::operator=(const ras_lipid_conduit_data_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void ras_lipid_conduit_data_reader::copy_members(const ras_lipid_conduit_data_reader &rhs) {
  if (is_master()) {
    std::cout << "Starting ras_lipid_conduit_data_reader::copy_members\n";
  }
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);
  m_filenames = rhs.m_filenames;
  m_samples_per_file = rhs.m_samples_per_file;
  m_data_id_map = rhs.m_data_id_map;
  m_datum_sizes = rhs.m_datum_sizes;
  m_datum_bytes = rhs.m_datum_bytes;
}

void ras_lipid_conduit_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load" << std::endl;
  }

  options *opts = options::get();

  if (! opts->get_bool("preload_data_store")) {
    LBANN_ERROR("ras_lipid_conduit_data_reader requires data_store; please pass either --preload_data_store on the cmd line");
  }

  //dah - for now, I assume the input file contains, on each line, the complete
  //      pathname of an npz file. 
  std::string infile = get_data_filename();
  read_filelist(m_comm, infile, m_filenames);

  // P_0 counts the number of samples in each file, then bcasts to
  // others. Also check that all arrays in a file have the same leading
  // dimension; this probably isn't necessary, but let's not take chances
  //TODO: make this distributed; for now, just make it work
  m_samples_per_file.reserve(m_filenames.size());
  if (m_comm->get_rank_in_trainer() == 0) {
    for (const auto &fn : m_filenames) {
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(fn);
      size_t n = 0;
      for (const auto &t2 : a) {
        size_t n2 = t2.second.shape[0];
        if (n == 0) {
          n = n2;
        } else {
          if (n2 != n) {
            LBANN_ERROR("n2 != n; ", n2, n);
          }
        }
      }
      m_samples_per_file.push_back(n);
    }
  }
  m_comm->trainer_broadcast<size_t>(0, m_samples_per_file.data(), m_samples_per_file.size());

  //Note: we really need the sample list here, but to get this working
  //I'm doing something clunky ...
  int data_id = 0;
  for (size_t j=0; j<m_samples_per_file.size(); j++) {
    for (size_t h=0; h<m_samples_per_file[j]; h++) {
      m_data_id_map[data_id++] = std::make_pair(j,h);
    }
  }

  // compute number of global samples, then setup the shuffled indices
  m_num_samples = 0;
  for (auto t : m_samples_per_file) {
    m_num_samples += t;
  }
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  m_num_samples = m_shuffled_indices.size();

  instantiate_data_store();
  select_subset_of_data();
}

void ras_lipid_conduit_data_reader::do_preload_data_store() {
  if (is_master()) std::cout << "Starting ras_lipid_conduit_data_reader::preload_data_store; num indices: " << m_shuffled_indices.size() << std::endl;
  double tm1 = get_time();

  // build the set of indices that this reader uses
  std::unordered_set<size_t> indices;
  for (const auto &t : m_shuffled_indices) {
    indices.insert(t);
  }

  // compute the data_id of the first sample in each npz file
  std::vector<size_t> first_samples(m_samples_per_file.size());
  first_samples[0] = 0;
  for (size_t h=0; h<m_samples_per_file.size()-1; h++) {
    first_samples[h+1] = first_samples[h] + m_samples_per_file[h];
  }

  int my_rank = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();

  // re-build the data store's owner map, and get the set of
  // data_ids that this rank owns
  m_data_store->clear_owner_map();
  std::unordered_map<int, std::vector<std::pair<int,int>>> my_samples;
  for (size_t j=0; j<m_filenames.size(); ++j) {
    int file_owner = j % np;
    size_t data_id = first_samples[j];
    for (size_t h=0; h<m_samples_per_file[j]; h++) {
      ++data_id;
      if (indices.find(data_id) != indices.end()) {
        m_data_store->add_owner(data_id, file_owner);
        if (file_owner == my_rank) {
          my_samples[j].push_back(std::make_pair(data_id, h));
        }
      }
    }
  }

  // get the shapes for the various arrays
  std::map<std::string, cnpy::NpyArray> aa = cnpy::npz_load(m_filenames[0]);
  std::unordered_map<std::string, std::vector<int>> shapes;
  std::unordered_map<std::string, int> sample_bytes;
  for (const auto &t : aa) {
    const std::string &name = t.first;
    const std::vector<size_t> &s = t.second.shape;
    size_t word_size = t.second.word_size;
    int num_bytes = 1;
    for (size_t i=1; i<s.size(); i++) {
      shapes[name].push_back(s[i]);
      num_bytes *= s[i];
    }
    num_bytes *= word_size;
    sample_bytes[name] = num_bytes;
  }

  // construct a conduit::Node for each sample that this rank owns,
  // and set it in the data_store
  for (const auto &t : my_samples) {
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(m_filenames[t.first]);
    for (const auto &t4 : t.second) {
      int data_id = t4.first;
      int offset = t4.second;
      conduit::Node &node = m_data_store->get_empty_node(data_id);

      for (const auto &t5 : m_datum_sizes) {
        const std::string &name = t5.first;
        conduit::uint8 *data = reinterpret_cast<conduit::uint8*>(a[name].data_holder->data());
        node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset*m_datum_bytes[name], m_datum_bytes[name]); 
      }
      m_data_store->set_conduit_node(data_id, node);
    }
  }  

  double tm2 = get_time();
  if (is_master()) {
    std::cout << "time to preload: " << tm2 - tm1 << " for role: " << get_role() << std::endl;
  }
}

bool ras_lipid_conduit_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
#if 0
  Mat X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));
  conduit::Node node;
  if (data_store_active()) {
    const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
    node.set_external(ds_node);
  } else {
    load_npz(m_filenames[data_id], data_id, node);
    //note: if testing, and test set is touched more than once, the following
    //      will throw an exception TODO: relook later
    const auto& c = static_cast<const execution_context&>(m_model->get_execution_context());
    if (priming_data_store() || c.get_execution_mode() == execution_mode::testing) {
      m_data_store->set_conduit_node(data_id, node);
    }
  }
  #endif

/*
  TODO
  const char *char_data = node[LBANN_DATA_ID_STR(data_id) + "/data/data"].value();
  char *char_data_2 = const_cast<char*>(char_data);
  void *data = (void*)char_data_2;
  std::memcpy(X_v.Buffer(), data, m_num_features * m_data_word_size);
*/

  return true;
}

bool ras_lipid_conduit_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  double label = node[LBANN_DATA_ID_STR(data_id) + "/states"].value();
  Y.Set(label, mb_idx, 1);
  return true;
}

bool ras_lipid_conduit_data_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("ras_lipid_conduit_data_reader: do not have responses");
  return true;
}

void ras_lipid_conduit_data_reader::fill_in_metadata() {
  std::map<std::string, cnpy::NpyArray> aa = cnpy::npz_load(m_filenames[0]);
  for (const auto &t : aa) {
    const std::string &name = t.first;
    size_t word_size = t.second.word_size;
    const std::vector<size_t> &shape = t.second.shape;
    size_t b = 1;
    for (size_t x=1; x<shape.size(); x++) {
      b *= shape[x];
      m_datum_sizes[name].push_back(shape[x]);
    }
    b *= word_size;
    m_datum_bytes[name] = b;
  }
}


}  // namespace lbann
