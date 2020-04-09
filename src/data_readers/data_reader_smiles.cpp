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
  opts->set_option("preload_data_store", 1);

  std::string infile = get_data_filename();

  // Open input data file; get num samples and max sample size
  m_data_stream.open(infile, std::ios::binary);
  if (!m_data_stream) {
    LBANN_ERROR("failed to open SMILES data file for reading: ", infile);
  }
  m_data_stream.seekg(0, m_data_stream.end);
  int len =  m_data_stream.tellg();
  len -= sizeof(int);
  m_data_stream.seekg(len);
  int num_samples;
  m_data_stream.read((char*)&num_samples, sizeof(int));
  len -= sizeof(short);
  m_data_stream.seekg(len);
  short max_sample_size;
  m_data_stream.read((char*)&max_sample_size, sizeof(short));
  m_linearized_data_size = max_sample_size;
  m_data_stream.seekg(0);

  m_sample_offsets.reserve(num_samples);
  m_sample_sizes.reserve(num_samples);
  short n_chars;
  for (int j=0; j<num_samples; j++) {
    m_data_stream.read((char*)&n_chars, sizeof(short));
    m_sample_sizes.push_back(n_chars);
    m_sample_offsets.push_back(m_data_stream.tellg());
  }

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  instantiate_data_store();
  select_subset_of_data();
}

void smiles_data_reader::do_preload_data_store() {
  if (is_master()) std::cout << "starting smiles_data_reader::do_preload_data_store; num indices: " << utils::commify(m_shuffled_indices.size()) << " for role: " << get_role() << std::endl;
  LBANN_ERROR("smiles_data_reader::do_preload_data_store is not yet implemented");
}

bool smiles_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  bool have_node = true;
  conduit::Node node;
  std::vector<short> data_vector;
  if (m_data_store != nullptr) {
    if (data_store_active()) {
      //get data from node from data store
      const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
      node.set_external(ds_node);
      have_node = true;
    } else if (priming_data_store()) {
      //get data from file, and stuff it in the data store;
      // data is retrieved in vector<short> data
      load_conduit_node_from_file(data_id, node);
      m_data_store->set_conduit_node(data_id, node);
      have_node = false;
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
    read_datum(data_id, data_vector);
  }

  short *v;
  if (have_node) {
    //get v* from conduit node
    v =  node[LBANN_DATA_ID_STR(data_id) + "/density_sig1"].value(); 
  } else {
    //get v* from vector<short> data
    v = data_vector.data();
  }
  int n = m_sample_sizes[data_id];
  int j;
  for (j = 0; j < n; ++j) {
    X(j, mb_idx) = v[j]; 
  }
  for (; j<m_linearized_data_size; j++) {
    X(j, mb_idx) = m_pad;
  }
  
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

void smiles_data_reader::read_datum(const int data_id, std::vector<short> &data_out) {
  //TODO: get rid of mutex
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_data_stream.is_open()) {
    LBANN_ERROR("input data stream is not open, but should be");
  }
  m_data_stream.seekg(m_sample_offsets[data_id]);
  short num_chars;
  m_data_stream.read((char*)&num_chars, sizeof(short));
  if (num_chars != m_sample_sizes[data_id]) {
    LBANN_ERROR("num_chars != m_sample_sizes[data_id] but should be");
  }
  data_out.resize(num_chars);
  m_data_stream.read((char*)data_out.data(), sizeof(short)*num_chars);
}

void smiles_data_reader::load_conduit_node_from_file(const int data_id, conduit::Node &node) {
  std::vector<short>  data;
  read_datum(data_id, data);
  node[LBANN_DATA_ID_STR(data_id) + "/data"].set(data);
}

std::vector<El::Int> smiles_data_reader::get_slice_points() const {
  std::vector<El::Int> p;
  for (int j=0; j<m_linearized_data_size; j++) {
    p.push_back(j);
  }
  return p;
}

}  // namespace lbann
