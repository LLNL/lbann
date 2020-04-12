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

#include "lbann/data_readers/data_reader_smiles.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/lbann_library.hpp"
#include <mutex>

namespace lbann {

smiles_data_reader::smiles_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

smiles_data_reader::smiles_data_reader(const smiles_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

smiles_data_reader::~smiles_data_reader() {
  if (m_data_stream.is_open()) {
    m_data_stream.close();
  }
}

smiles_data_reader& smiles_data_reader::operator=(const smiles_data_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void smiles_data_reader::copy_members(const smiles_data_reader &rhs) {
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_data_store->set_data_reader_ptr(this);
  m_linearized_data_size = rhs.m_linearized_data_size;
  m_linearized_label_size = rhs.m_linearized_label_size;
  m_linearized_response_size = rhs.m_linearized_response_size;
  m_num_labels = rhs.m_num_labels;
}

void smiles_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();
  if (!opts->has_int("preload_data_store")) {
    if (is_master()) {
      LBANN_WARNING("setting --preload_data_store");
    }  
    opts->set_option("preload_data_store", 1);
  }

  // load the vocabulary; this is a map: string -> short
  load_vocab();

  // get the number of samples to use; if no param: --num_samples,
  // then the entire file will be read. In this case we count the
  // number of lines in the file, which may be slow for large files.
  // Or maybe not.
  std::string infile = get_data_filename();
  int num_samples = get_num_samples();
  if (opts->has_int("num_samples")) {
    num_samples = opts->get_int("num_samples");
  }
  /*
  if (num_samples == -1) {
    num_samples = get_num_lines(infile);
  }
  */

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  instantiate_data_store();
  select_subset_of_data();
}

void smiles_data_reader::do_preload_data_store() {
  if (is_master()) {
    std::cout << "starting smiles_data_reader::do_preload_data_store; num indices: " 
              << utils::commify(m_shuffled_indices.size()) 
              << " for role: " << get_role() << std::endl;
  }

  const std::string infile = get_file_dir() + "/" + get_data_filename();
  std::ifstream in(infile.c_str());
  if (!in) {
    LBANN_ERROR("failed to open data file: ", infile, " for reading");
  }
  std::string line;
  getline(in, line); //assume a header line, and discard

  //TODO: this is terrible! Maybe: have root scan the file and bcast a map:
  //      offset->line number
  int rank = m_comm->get_rank_in_trainer();
  conduit::Node node;
  for (size_t data_id=0; data_id<m_shuffled_indices.size(); data_id++) {
    getline(in, line);
    int index = m_shuffled_indices[data_id];
    if (m_data_store->get_index_owner(index) != rank) {
      continue;
    }
    construct_conduit_node(index, line, node);
    m_data_store->set_preloaded_conduit_node(index, node);
  }
  in.close();
}

bool smiles_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  bool have_node = true;
  conduit::Node node;
  if (m_data_store != nullptr) {
    if (data_store_active()) {
      //get data from node from data store
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
      have_node = true;
    } else if (priming_data_store()) {
      //get data from file, and stuff it in the data store;
      // data is retrieved in vector<short> data
      have_node = false;
      LBANN_ERROR("explicit loading not currently supported; it should be impossible to be here");
    } else {
      have_node = false;
      if (get_role() != "test") {
        LBANN_ERROR("using data_store for smiles_data_reader::fetch_datum is not implemented for role=test (actually, you probably shouldn't be here; please contact Dave Hysom)");
      } else {
       LBANN_ERROR("data_store is active, but data_store_active()=false and priming_data_store() = false. It should be impossible to be here; please contact Dave Hysom");
      }
    }
  }

  // this block fires if not using data store
  else {
     LBANN_ERROR("smiles_data_reader requires the data_store; this may change in the future");
    //read_datum(data_id, data_vector);
  }

  short *v;
  int n = 0;
  if (have_node) {
    //get v* from conduit node
    v =  node[LBANN_DATA_ID_STR(data_id) + "/data"].value(); 
    n = node[LBANN_DATA_ID_STR(data_id) + "/size"].value();
  } else {
    //TODO
  }
  int j;
  for (j = 0; j < n; ++j) {
    X(j, mb_idx) = v[j]; 
  }
  /*
  for (; j<m_linearized_data_size; j++) {
    X(j, mb_idx) = m_pad;
  }
  */
  
  return true;
}

bool smiles_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("smiles_data_reader::fetch_label is not implemented");
  return true;
}

bool smiles_data_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("smiles_data_reader::fetch_response is not implemented");
  return true;
}


//user feedback
void smiles_data_reader::print_statistics() const {
  if (!is_master()) {
    return;
  }

  std::cout << "\n======================================================\n";
#if 0
  std::cout << "num train samples=" << m_num_train_samples << std::endl;
  std::cout << "num validate samples=" << m_num_validate_samples << std::endl;
  std::cout << "sequence length=" << m_seq_len << std::endl;
  std::cout << "num features=" << get_linearized_data_size() << std::endl;
  std::cout << "num labels=" << get_num_labels() << std::endl;
  std::cout << "data dims=";
#endif
  std::cout << "======================================================\n\n";
}

void smiles_data_reader::load_vocab() {
  std::vector<char> v { 
    '#', '%', '(', ')', '+', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', 
    '8', '9', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', ']', 
    'e', 'i', 'l', 'm', 'r', 's'};

  short j = 0;
  for (const auto &t : v) {
    m_vocab[t] = j++;
  }
/*
  options *opts = options::get();
  if (!opts->has_string("vocab_fn")) {
    LBANN_ERROR("you must pass: --vocab_fn=<string>");
  }
  const std::string fn = opts->get_string("vocab_fn");
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open vocab file: ", fn, " for reading");
  }
  std::string key;
  short value;
  while (in >> key >> value) {
    m_vocab[key = value];
  }
  in.close();
*/
}

int smiles_data_reader::get_num_lines(std::string fn) {
  //TODO: master should read and bcast; notify usr every n lines
  //      May be faster to do the canonical C thing
  double tm1 = get_time();
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open date file: ", fn, " for reading");
  }
  std::string line;
  int count = 0;
  while(getline(in,line)) {
    ++count;
  }
  in.close();

  if (is_master()) {
    std::cout << "smiles_data_reader::get_num_lines; num_lines: " 
              << count << " time: " << get_time()-tm1 << std::endl;
  }
  return count;
}

void smiles_data_reader::construct_conduit_node(int data_id, const std::string &line, conduit::Node &node) {
  node.reset();
  size_t j = line.find('\t');
  if (j == std::string::npos) {
    LBANN_ERROR("failed to find tab character in line: ", line);
  }
  std::string sm = line.substr(0, j);
  std::vector<short> data;
  encode_smiles(sm, data);
//XXsize_t s1 = sm.size();
  node[LBANN_DATA_ID_STR(data_id) + "/data"].set(data);
  int sz = data.size();

  //XXstd::cout << "=== starting construct_conduit_node; data_id: " << data_id << std::endl;
  //XXstd::cout << "smiles: " << s1 << " final size: " << sz << std::endl;
  //XXstd::cout << sm << std::endl;
  node[LBANN_DATA_ID_STR(data_id) + "/size"].set(sz);
}

void smiles_data_reader::encode_smiles(const std::string &sm, std::vector<short> &data) {
  //TODO: would training be better with an algorithm that deals 
  //      with multi-character tokens?
  int stop = sm.size();
  static int count = 0;

  if (stop > m_linearized_data_size) {
    stop = m_linearized_data_size;
    if (is_master()) {
      if (count < 20) {
        count += 1;
        LBANN_WARNING("smiles string size is ", sm.size(), "; losing ", (sm.size()-m_linearized_data_size), " characters");
      }
    }
  }

  for (int j=0; j<stop; j++) {
    if (m_vocab.find(sm[j]) == m_vocab.end()) {
      data.push_back(m_unk);
    } else {
      data.push_back(m_vocab[sm[j]]);
    }
  }
  for (size_t j = data.size(); j<(size_t)m_linearized_data_size; j++) {
    data.push_back(m_pad);
  }
}


}  // namespace lbann
