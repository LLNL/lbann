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
  m_datum_shapes = rhs.m_datum_shapes;
  m_datum_word_sizes = rhs.m_datum_word_sizes;
  m_datum_num_bytes = rhs.m_datum_num_bytes;
  m_datum_num_words = rhs.m_datum_num_words;
  m_num_features = rhs.m_num_features;
  m_num_labels = rhs.m_num_labels;
  m_num_response_features = rhs.m_num_response_features;
  m_data_dims = rhs.m_data_dims;
}

void ras_lipid_conduit_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();

  if (! opts->get_bool("preload_data_store")) {
    LBANN_ERROR("ras_lipid_conduit_data_reader requires data_store; please pass either --preload_data_store on the cmd line");
  }

  //dah - for now, I assume the input file contains, on each line, the complete
  //      pathname of an npz file. 
  std::string infile = get_data_filename();
  read_filelist(m_comm, infile, m_filenames);

  fill_in_metadata();

  if (opts->has_string("pilot2_read_file_sizes")) {
    read_file_sizes();
  } else {
    double tm3 = get_time();
    get_samples_per_file();
    if (is_master()) std::cout << "time to compute samples_per_file: " << get_time() - tm3 << std::endl;
  }  
  if (opts->has_string("pilot2_save_file_sizes")) {
    write_file_sizes();
  }

  //Note: we really need the sample list here, but to get this working
  //I'm doing something clunky ...
  //XX
  /*
  int data_id = 0;
  for (size_t j=0; j<m_samples_per_file.size(); j++) {
    for (size_t h=0; h<m_samples_per_file[j]; h++) {
      m_data_id_map[data_id++] = std::make_pair(j,h);
    }
  }
  */

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
  if (is_master()) std::cout << "Starting ras_lipid_conduit_data_reader::do_preload_data_store; num indices: " << m_shuffled_indices.size() << std::endl;

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

  // two hacks follow. I call them hack, because they wouldn't be needed
  // if we were using sample lists. 

  // hack: re-build the data store's owner map
  rebuild_data_store_owner_map();

  // hack: get the data_ids that this rank owns
  std::unordered_map<int, std::vector<std::pair<int,int>>> my_samples;
  get_my_indices(my_samples);

  // construct a conduit::Node for each sample that this rank owns,
  // and set it in the data_store
  //bool print_stats = true;
size_t nn = 0;
  for (const auto &t : my_samples) {
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(m_filenames[t.first]);
    for (const auto &t4 : t.second) {
      int data_id = t4.first;
      int sample_index = t4.second;
      conduit::Node &node = m_data_store->get_empty_node(data_id);

      size_t offset;
      for (const auto &t5 : m_datum_shapes) {
        const std::string &name = t5.first;
        // this could be done better ... read the choices of fields
        // to use from file, as is done in data_reader_jag_conduit?

        if (name == "frames") {
          conduit::int64 *data = reinterpret_cast<conduit::int64*>(a[name].data_holder->data());
          offset = sample_index*m_datum_num_words["frames"];
          node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
        } 

        else if (name == "bbs") {
          conduit::float32 *data = reinterpret_cast<conduit::float32*>(a[name].data_holder->data());
          offset = sample_index*m_datum_num_words["bbs"];
          node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
        } 

        else { // rots, states, tilts, density_sig1, probs
          offset = sample_index*m_datum_num_words[name];
          conduit::float64 *data = reinterpret_cast<conduit::float64*>(a[name].data_holder->data());
          if (name == "states") {
            int label = (data + offset)[0];
            node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(label);
          } else {
            node[LBANN_DATA_ID_STR(data_id) + "/" + name].set(data + offset, m_datum_num_words[name]);
          }  
        }
      }
      m_data_store->set_preloaded_conduit_node(data_id, node);
      ++nn;
    }
  }  
}

bool ras_lipid_conduit_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  //TODO: compute normalization (scaling factor)
  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  double scaling_factor = 1.0;
  const double *data = node[LBANN_DATA_ID_STR(data_id) + "/density_sig1"].value();
  size_t n = m_datum_num_words["density_sig1"];
  for (size_t j = 0; j < n; ++j) {
    X(j, mb_idx) = data[j] * scaling_factor;
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

std::map<double,int> m2;

bool ras_lipid_conduit_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  const conduit::Node node = m_data_store->get_conduit_node(data_id);
  int label = node[LBANN_DATA_ID_STR(data_id) + "/states"].value();
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
    size_t num_words = 1;
    if (shape.size() == 1) {
      m_datum_shapes[name].push_back(1);
    } else {
      for (size_t x=1; x<shape.size(); x++) {
        num_words *= shape[x];
        m_datum_shapes[name].push_back(shape[x]);
      }
    }
    m_datum_num_words[name] = num_words;
    m_datum_word_sizes[name] = word_size;
    m_datum_num_bytes[name] = num_words*word_size;

  }

  //TODO: this should be more generic, will need to change depending on what we fetch
  if (m_datum_shapes.find("density_sig1") == m_datum_shapes.end()) {
    LBANN_ERROR("m_datum_shapes.find(\"density_sig1\") = m_datum_shapes.end()");
  }
  for (auto t : m_datum_shapes["density_sig1"]) {
    m_data_dims.push_back(t);
  }
}

void ras_lipid_conduit_data_reader::get_my_indices(std::unordered_map<int, std::vector<std::pair<int,int>>> &my_samples) {
  std::unordered_set<size_t> indices;
  for (const auto &t : m_shuffled_indices) {
    indices.insert(t);
  }
  int my_rank = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();
  size_t data_id = 0;
  for (size_t j=0; j<m_filenames.size(); ++j) {
int x = 0;
    int file_owner = j % np;
    for (int h=0; h<m_samples_per_file[j]; h++) {
      if (indices.find(data_id) != indices.end()) {
        if (file_owner == my_rank) {
          my_samples[j].push_back(std::make_pair(data_id, h));
++x;
        }
      }
      ++data_id;
    }
  }
}

void ras_lipid_conduit_data_reader::rebuild_data_store_owner_map() {
  m_data_store->clear_owner_map();
  int np = m_comm->get_procs_per_trainer();
  size_t data_id = 0;
  for (size_t j=0; j<m_filenames.size(); ++j) {
    int file_owner = j % np;
    for (int h=0; h<m_samples_per_file[j]; h++) {
      m_data_store->add_owner(data_id, file_owner);
      ++data_id;
    }
  }
}

void ras_lipid_conduit_data_reader::get_samples_per_file() {
  int me = m_comm->get_rank_in_trainer();
  int np = m_comm->get_procs_per_trainer();
  std::vector<int> work;
  int x = 0;
  for (size_t j=me; j<m_filenames.size(); j+=np) {
    ++x;
    std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(m_filenames[j]);
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
    work.push_back(j);
    work.push_back(n);
  }

  std::vector<int> num_files(np, 0);
  for (size_t j=0; j<m_filenames.size(); ++j) {
    int owner = j % np;
    num_files[owner] += 1;
  }

  m_samples_per_file.resize(m_filenames.size());
  std::vector<int> work_2;
  std::vector<int> *work_ptr;
  for (int j=0; j<np; j++) {
    if (me == j) {
      work_ptr = &work;
    } else {
      work_2.resize(num_files[j]*2);
      work_ptr = &work_2;
    }
    m_comm->trainer_broadcast<int>(j, work_ptr->data(), work_ptr->size());
    for (size_t h=0; h<work_ptr->size(); h+= 2) {
      m_samples_per_file[(*work_ptr)[h]] = (*work_ptr)[h+1];
    }
  }
}

void ras_lipid_conduit_data_reader::write_file_sizes() {
  if (! is_master()) {
    return;
  }
  std::string fn = options::get()->get_string("pilot2_save_file_sizes");
  std::ofstream out(fn.c_str());
  if (!out) {
    LBANN_ERROR("failed to open ", fn, " for writing");
  }
  for (size_t j=0; j<m_samples_per_file.size(); j++) {
    out << m_filenames[j] << " " << m_samples_per_file[j] << std::endl;
  }
  out.close();
}

void ras_lipid_conduit_data_reader::read_file_sizes() {
if (is_master()) std::cout << "starting ras_lipid_conduit_data_reader::read_file_sizes\n";
  std::string fn = options::get()->get_string("pilot2_read_file_sizes");
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", fn, " for reading");
  }
  std::unordered_map<std::string, int> mp;
  std::string filename;
  int num_samples;
  while (in >> filename >> num_samples) {
    mp[filename] = num_samples;
  }
  in.close();

  m_samples_per_file.resize(m_filenames.size());
  for (size_t h=0; h<m_filenames.size(); h++) {
    if (mp.find(m_filenames[h]) == mp.end()) {
      LBANN_ERROR("failed to find filename '", m_filenames[h], "' in the map");
    }
    m_samples_per_file[h] = mp[m_filenames[h]];
  }
}

}  // namespace lbann
