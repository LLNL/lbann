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
  if (m_missing_chars.size()) {
    if (is_master()) {
      std::cout << std::endl << "Tokens in data that were missing from vocab: ";
      for (const auto t : m_missing_chars) {
        std::cout << t << " ";
      }
    }
    std::cout << std::endl;
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
  m_num_samples = rhs.m_num_samples;
  m_data_store->set_data_reader_ptr(this);
  m_linearized_data_size = rhs.m_linearized_data_size;
  m_linearized_label_size = rhs.m_linearized_label_size;
  m_linearized_response_size = rhs.m_linearized_response_size;
  m_num_labels = rhs.m_num_labels;
  m_pad = rhs.m_pad;
  m_unk = rhs.m_unk;
  m_bos = rhs.m_bos;
  m_eos = rhs.m_eos;
  m_missing_char_in_vocab = rhs.m_missing_char_in_vocab;
  m_missing_chars = rhs.m_missing_chars;
  m_fast_experimental = rhs.m_fast_experimental;
}

void smiles_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();

  m_fast_experimental = false;
  if (options::get()->get_bool("fast_experimental")) {
    m_fast_experimental = true;
    if (is_master()) {
      std::cerr << "\nSMILES_DATA_READER is running in --fast_experimental mode\n";
    }
  } else { // run in normal --preload_data_store mode
    if (!opts->has_int("preload_data_store")) {
      if (is_master()) {
        LBANN_WARNING("setting --preload_data_store");
      }
      opts->set_option("preload_data_store", 1);
    }
  }

  if (!opts->has_int("sequence_length")) {
    LBANN_ERROR("you must pass --sequence_length=<int> on the cmd line");
  }
  m_linearized_data_size = opts->get_int("sequence_length");

  // load the vocabulary; this is a map: string -> short
  int sanity = load_vocab();
  if (!(opts->has_int("num_embeddings") && opts->has_int("embedding_dim"))) {
    LBANN_ERROR("you must pass --num-embeddings=<int> and --embedding-dim=<int> on the cmd line");
  }
  int n_embeddings = opts->get_int("num_embeddings");
  int n_embedding_dim = opts->get_int("embedding_dim");
  if (sanity != n_embeddings || sanity != n_embedding_dim) {
    LBANN_ERROR("--num_embeddings=", n_embeddings, "; --embedding_dim=", n_embedding_dim, "; both should be the same as vocab_size which is: ", m_vocab.size());
  }

  // get the number of samples to use; if --num_samples=<int> wasn't 
  // passed on the cmd line, then the entire file will be used. In this case 
  // we count the number of lines in the file, which may be slow 
  // for large files. Or maybe not.
  if (opts->has_int("num_samples")) {
    m_num_samples = opts->get_int("num_samples");
  } else {
    const std::string infile = get_file_dir() + "/" + get_data_filename();
    m_num_samples = get_num_lines(infile) -1;
  }

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  instantiate_data_store();
  select_subset_of_data();

  if (m_fast_experimental) {
    setup_fast_experimental();
  }
}

void smiles_data_reader::do_preload_data_store() {
  if (is_master()) {
    std::cout << "starting do_preload_data_store; num indices: " 
              << utils::commify(m_shuffled_indices.size()) 
              << " for role: " << get_role() << std::endl;
  }
  m_data_store->set_node_sizes_vary();
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
  if (m_fast_experimental) {
    std::vector<short> data;
    get_sample(data_id, data);

    size_t j;
    for (j = 0; j < data.size(); ++j) {
      X(j, mb_idx) = data[j]; 
    }
    for (int jj = j; jj<m_linearized_data_size; jj++) {
      X(jj, mb_idx) = m_pad;
    }
  }

  // run with data_store in preload mode
  else {
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
    for (; j<m_linearized_data_size; j++) {
      X(j, mb_idx) = m_pad;
    }
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

int smiles_data_reader::load_vocab() {
  options *opts = options::get();
  if (!opts->has_string("vocab")) {
    LBANN_ERROR("you must pass --vocab=<string> on the command line");
  }
  const std::string fn = opts->get_string("vocab");
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", fn, " for reading; this is the vocabulary file");
  }
  std::string token;
  short id;
  int sanity = 4;
  while (in >> token >> id) {
    m_vocab[token] = id;
    if (token == "<pad>") {
      m_pad = id;
      --sanity;
    }
    if (token == "<unk>") {
      m_unk = id;
      --sanity;
    }
    if (token == "<bos>") {
      m_bos = id;
      --sanity;
    }
    if (token == "<eos>") {
      m_eos = id;
      --sanity;
    }
  }
  in.close();
  if (sanity) {
    LBANN_ERROR("failed to find <pad> and/or <unk> and/or <bos> and/or <eos> in vocab file: ", fn);
  }
  if (opts->has_int("pad_index")) {
    short tmp = opts->get_int("pad_index");
    if (tmp != m_pad) {
      LBANN_ERROR("you passed --pad_index=", tmp, " but we got --pad_index=", m_pad, " from the vocabulary file");
    }
  }
  return(m_vocab.size());
  
#if 0
  //, '<bos>': 34, '<eos>': 35, '<pad>': 36, '<unk>': 37}; 
  //
  //
//from Derek:
  //m_vocab = {'#': 0, '%': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '0': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, '=': 17, '@': 18, 'B': 19, 'C': 20, 'F': 21, 'H': 22, 'I': 23, 'N': 24, 'O': 25, 'P': 26, 'S': 27, '[': 28, ']': 29, 'e': 30, 'i': 31, 'l': 32, 'r': 33, '<bos>': 34, '<eos>': 35, '<pad>': 36, '<unk>': 37}; 
  
//mine:
//  std::vector<char> v { 
 //   '#', '%', '(', ')', '+', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', ']', 'e', 'i', 'l', 'm', 'r', 's'};
#endif
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
    j = line.find(',');
  }
  if (j == std::string::npos) {
    LBANN_ERROR("failed to find delimit character (tab or comma) in line: ", line);
  }
  std::string sm = line.substr(0, j);
  std::vector<short> data;
  encode_smiles(sm, data);
  node[LBANN_DATA_ID_STR(data_id) + "/data"].set(data);
  int sz = data.size();
  node[LBANN_DATA_ID_STR(data_id) + "/size"].set(sz);
}

void smiles_data_reader::encode_smiles(const std::string &sm, std::vector<short> &data) {
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
    const std::string w(1, sm[j]);
    if (m_vocab.find(w) == m_vocab.end()) {
      m_missing_chars.insert(w);
      ++m_missing_char_in_vocab;
      if (m_missing_char_in_vocab < 2) {
        LBANN_WARNING("smiles_data_reader::encode_smiles encounted character not in vocab: ", w, "; for SMILES string: ", sm);
      }  
      data.push_back(m_unk);
    } else {
      data.push_back(m_vocab[w]);
    }
  }
  /*
  for (size_t j = data.size(); j<(size_t)m_linearized_data_size; j++) {
    data.push_back(m_pad);
  }
  */
}

void smiles_data_reader::get_sample(int sample_id, std::vector<short> &sample_out) {
  std::unordered_map<int, std::pair<size_t, short>>::const_iterator iter = m_sample_lookup.find(sample_id);
  if (iter == m_sample_lookup.end()) {
    std::stringstream s;
    s << "; m_sample_lookup.size: " << m_sample_lookup.size() << " known data_ids: ";
    for (auto t : m_sample_lookup) s << t.first << " ";
    LBANN_ERROR("failed to find data_id ", sample_id, " in m_sample_lookup", s.str());
  }
  size_t offset = iter->second.first;
  short size = iter->second.second;

  if (offset + size > m_data.size()) {
    LBANN_ERROR("offset: ", offset, " + size: ", size, " is > m_data.size(): ", m_data.size());
  }

  const char *v = m_data.data()+offset;
  const std::string smiles_string(v, size);
  encode_smiles(smiles_string, sample_out);
}

// Let the hacking begin ...
// TODO: break into several function calls
// TODO: some/most of the following could/should be in the data_store ??
void smiles_data_reader::setup_fast_experimental() {
  std::vector<size_t> sample_offsets(m_shuffled_indices.size()*3);
  size_t buffer_size;

  if (is_master()) {
    std::cout << "\nSTARTING smiles_data_reader::setup_fast_experimental() " << std::endl << std::endl;
    double tm1 = get_time();

    // Open input file and discard header line
    const std::string infile = get_file_dir() + "/" + get_data_filename();
    std::ifstream in(infile.c_str());
    if (!in) {
      LBANN_ERROR("failed to open data file: ", infile, " for reading");
    }
    std::string line;
    getline(in, line); //assume a header line, and discard

    // Count memory requirements (ugh; possibly precompute);
    // This can be done better, but doing it the easy way for now;
    // do it the better way only if this is too slow
    std::unordered_set<int> samples_to_use;
    int max_sample_id = 0; //stop compiler complaints
    for (size_t j=0; j<m_shuffled_indices.size(); j++) {
      samples_to_use.insert(m_shuffled_indices[j]);
      max_sample_id = m_shuffled_indices[j] > max_sample_id ? m_shuffled_indices[j] : max_sample_id;
    }
    ++max_sample_id;

    sample_offsets.clear();
    size_t offset = 0;
    for (int j=0; j<max_sample_id; j++) {
      getline(in, line);
      if (samples_to_use.find(j) != samples_to_use.end()) {
        size_t k = line.find('\t');
        if (k == std::string::npos) {
          k = line.find(',');
        }
        if (k == std::string::npos) {
          LBANN_ERROR("failed to find delimit character (tab or comma) in line: ", line, " which is line number ", j);
        }
        sample_offsets.push_back(j);
        sample_offsets.push_back(offset);
        sample_offsets.push_back(k);
        offset += k;
      }
    }
    buffer_size = offset;
    m_data.resize(buffer_size);

    // Fill in the data buffer
    in.seekg(0);
    getline(in, line); //assume a header line, and discard
    offset = 0;
    for (int j=0; j<max_sample_id; j++) {
      getline(in, line);
      if (samples_to_use.find(j) != samples_to_use.end()) {
        size_t k = line.find('\t');
        if (k == std::string::npos) {
          k = line.find(',');
        }
        for (size_t n=0; n<k; n++) {
          m_data[n+offset] = line[n];
        }
        offset += k;
      }
    }

    std::cout << "Time for computing sample sizes: " << get_time() - tm1 << std::endl;
    if (sample_offsets.size()/3 != m_shuffled_indices.size()) {
      LBANN_ERROR("sample_offsets.size()/3: ", sample_offsets.size()/3, " should be equal to m_shuffled_indices.size which is ", m_shuffled_indices.size());
    }
  }

  m_comm->broadcast<size_t>(0, sample_offsets.data(), sample_offsets.size(), m_comm->get_world_comm());

  // Construct lookup table
  for (size_t j=0; j<sample_offsets.size(); j += 3) {
    m_sample_lookup[sample_offsets[j]] = 
      std::make_pair(sample_offsets[j+1], sample_offsets[j+3]);
  }

  // Bcast the sample buffer
  m_comm->broadcast<size_t>(0, &buffer_size, 1, m_comm->get_world_comm());
  m_data.resize(buffer_size);
  m_comm->broadcast<char>(0, m_data.data(), m_data.size(), m_comm->get_world_comm());
}

}  // namespace lbann
