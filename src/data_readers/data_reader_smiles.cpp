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
#include <random>
#include <time.h>

namespace lbann {

smiles_data_reader::smiles_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

smiles_data_reader::smiles_data_reader(const smiles_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

smiles_data_reader::~smiles_data_reader() {
  if (m_missing_chars.size()) {
    if (is_master()) {
      std::cout << std::endl << "The following tokens were in SMILES strings, but were missing from the vocabulary: ";
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
  m_data_store = nullptr;
  if (rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
      m_data_store->set_data_reader_ptr(this);
  }
  m_linearized_data_size = rhs.m_linearized_data_size;
  m_linearized_label_size = rhs.m_linearized_label_size;
  m_linearized_response_size = rhs.m_linearized_response_size;
  m_num_labels = rhs.m_num_labels;
  m_pad = rhs.m_pad;
  m_unk = rhs.m_unk;
  m_bos = rhs.m_bos;
  m_eos = rhs.m_eos;
  m_has_header = rhs.m_has_header;
  m_delimiter = rhs.m_delimiter;
  m_missing_char_in_vocab_count = rhs.m_missing_char_in_vocab_count;
  m_missing_chars = rhs.m_missing_chars;
  m_vocab = rhs.m_vocab;
  m_sample_lookup_map = rhs.m_sample_lookup_map;
}

void smiles_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();

  if (!opts->has_int("sequence_length")) {
    LBANN_ERROR("you must pass --sequence_length=<int> on the cmd line");
  }
  m_linearized_data_size = opts->get_int("sequence_length") +2;

  // load the vocabulary; this is a map: string -> short
  load_vocab();

  // m_has_header = !opts->get_bool("no_header");
  //  side effects -- hard code for now, relook later
  m_has_header = true;

  // get the total number of samples in the file
  const std::string infile = get_file_dir() + "/" + get_data_filename();
  int num_samples = get_num_lines(infile);
  if (m_has_header) {
    --num_samples;
  }

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  // Optionally run "poor man's" LTFB
  if (opts->get_bool("ltfb")) {
    if (is_master()) {
      std::cout << "running poor man's LTFB\n";
    }
    size_t my_trainer = m_comm->get_trainer_rank();
    size_t num_trainers = m_comm->get_num_trainers();
    std::set<int> my_trainers_indices;

    // Use two loops here, to assure all trainers have
    // the same number of samples
    // ensure then number of samples is evenly divisible by
    // the number of trainers
    size_t n = m_shuffled_indices.size() / num_trainers;
    size_t s3 = n*num_trainers;
    if (m_shuffled_indices.size() != s3) {
      if (is_master()) {
        std::cout << "adjusting global sample size from " << m_shuffled_indices.size() << " to " << s3 << std::endl;
      }
      m_shuffled_indices.resize(s3);
    }
    for (size_t j=0; j<m_shuffled_indices.size(); j += num_trainers) {
      for (size_t k=0; k<num_trainers; k++) {
        int idx = j+k;
        if (idx % num_trainers == my_trainer)
          my_trainers_indices.insert(m_shuffled_indices[idx]);
      }
    }

    m_shuffled_indices.clear();
    for (const auto &t : my_trainers_indices) {
      m_shuffled_indices.push_back(t);
    }

  } else {
    if (is_master()) std::cout << "NOT running ltfb\n";
  }

  instantiate_data_store();
  select_subset_of_data();

  get_delimiter();

  if (m_data_store == nullptr) {
    build_sample_lookup_map();
  }
  print_statistics();
}

void smiles_data_reader::do_preload_data_store() {
  double tm1 = get_time();
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
  if (m_has_header) {
    getline(in, line);
  }

  // Collect the (global) set of sample_ids to be used in this experiment
  std::unordered_set<int> valid_ids;
  int sanity_min = INT_MAX;
  int sanity_max = 0;
  for (size_t idx=0; idx<m_shuffled_indices.size(); idx++) {
    int id = m_shuffled_indices[idx];
    if (id < sanity_min) sanity_min = id;
    if (id > sanity_max) sanity_max = id;
    if (m_data_store->get_index_owner(id) != m_comm->get_rank_in_trainer()) {
      continue;
    }
    valid_ids.insert(id);
  }

  int sample_id = -1;
  size_t sanity = 0;
  while (true) {
    ++sample_id;
    getline(in, line);
    if (valid_ids.find(sample_id) != valid_ids.end()) {
      if(m_data_store->get_index_owner(sample_id) != m_rank_in_model) {
        continue;
      }

      conduit::Node &node = m_data_store->get_empty_node(sample_id);
      construct_conduit_node(sample_id, line, node);
      m_data_store->set_preloaded_conduit_node(sample_id, node);
      ++sanity;
    }
    if (sample_id >= max_index) {
      break;
    }
    if (is_master() && (sanity % 1000000 == 0) && sanity > 0) {
      std::cout << sanity/1000000 << "M " << get_role() << " samples loaded" << std::endl;
    }
  }
  in.close();
  m_data_store->set_loading_is_complete();

  // Sanity check
  if (sanity != valid_ids.size()) {
    LBANN_ERROR("sanity != valid_ids.size() (sanity=", sanity, "; valid_ids.size()=", valid_ids.size());
  }

  if (is_master()) {
    std::cout << " do_preload_data_store time: " << get_time() - tm1 << std::endl;
  }
}

bool smiles_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  short *data_ptr = nullptr;
  size_t sz = 0;
  std::vector<short> data;
  // read sample from file
  if (m_data_store == nullptr) {
    get_sample(data_id, data);
    data_ptr = data.data();
    sz = data.size();
  }

  // get sample from the data_store
  else {
    if (! data_store_active()) {
      LBANN_ERROR("it should be impossible you you to be here; please contact Dave Hysom");
    }

    //get data from node from data store
    const conduit::Node& node = m_data_store->get_conduit_node(data_id);
    const std::string &smiles_string = node["/" + LBANN_DATA_ID_STR(data_id) + "/data"].as_string();
    encode_smiles(smiles_string, data, data_id);
    data_ptr = data.data();
    sz = data.size();
  }

  size_t j;
  for (j = 0; j < sz; ++j) {
    X(j, mb_idx) = data_ptr[j];
  }
  for (; j<static_cast<size_t>(m_linearized_data_size); j++) {
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
  std::cout << "role: " << get_role() << std::endl;
  //std::cout << "mem for data, lower bound: " << utils::commify(get_mem_usage()) << std::endl;
  std::cout << "num samples per trainer: " << m_shuffled_indices.size() << std::endl;
  std::cout << "max sequence length: " << utils::commify(m_linearized_data_size) << std::endl;
  std::cout << "num features=" << utils::commify(m_linearized_data_size) << std::endl;
  if (m_delimiter == '\t') {
    std::cout << "delimiter: <tab>\n";
  } else if (m_delimiter == ',') {
    std::cout << "delimiter: <comma>\n";
  } else if (m_delimiter == '\0') {
    std::cout << "delimiter: <none>\n";
  } else {
    LBANN_ERROR("invalid delimiter character, as int: ", (int)m_delimiter);
  }
  std::cout << "pad index: " << m_pad << std::endl;

  // +4 for <bos>, <eos>, <unk>, <pad>
  std::cout << "vocab size: " << m_vocab.size() +4 << std::endl
            << "    (includes +4 for <bos>, <eos>, <pad>, <unk>)" << std::endl
            << "======================================================\n\n";
}

void smiles_data_reader::load_vocab() {
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
    if (token.size() == 1) {
      m_vocab[token[0]] = id;
      m_vocab_inv[id] = token[0];
    }
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
}

int smiles_data_reader::get_num_lines(std::string fn) {
  double tm1 = get_time();
  int count = 0;
  if (is_master()) {
    std::ifstream in(fn.c_str());
    if (!in) {
      LBANN_ERROR("failed to open data file: ", fn, " for reading");
    }
    std::cout << "opened " << fn << " for reading\n";
    std::string line;
    while(getline(in,line)) {
      ++count;
    }
    in.close();

    std::cout << "smiles_data_reader::get_num_lines; num_lines: "
              << utils::commify(count) << " time: " << get_time()-tm1 << std::endl;
  }

  //I'm putting temporary timing around the bcast, because it
  //seems to be taking a long time
  if (is_master()) std::cout << "XX calling bcast ..." << std::endl;
  tm1 = get_time();
  m_comm->broadcast<int>(0, &count, 1, m_comm->get_world_comm());
  double tm = get_time() - tm1;
  if (is_master()) std::cout << "XX DONE! calling bcast ... TIME: " << tm << std::endl;

  //check if user want less than all samples in this file
  //@todo, this (flag or entire function) should really be deprecated since it can be accomplished with absoulte sample count
  options *opts = options::get();
  int n_lines = INT_MAX;
  if (opts->has_int("n_lines")) {
     n_lines = opts->get_int("n_lines");
     if(is_master() && count < n_lines) {
       std::cout << "WARNING:: number of available samples (" << count
                << " ) in file " << fn << " is less than number of samples requested (" << n_lines
                << " ) I am returning number of available samples " << std::endl;
       }
  }
  return std::min(count,n_lines);
}

void smiles_data_reader::construct_conduit_node(int data_id, const std::string &line, conduit::Node &node) {
  node.reset();
  int sz = get_smiles_string_length(line, data_id);
  const std::string sm = line.substr(0, sz);
  node[LBANN_DATA_ID_STR(data_id) + "/data"] = sm;
}

void smiles_data_reader::encode_smiles(const std::string &smiles, std::vector<short> &data, int data_id) {
  encode_smiles(smiles.data(), smiles.size(), data, data_id);
}

void smiles_data_reader::encode_smiles(const char *smiles, short size, std::vector<short> &data, int data_id) {
  static int count = 0;

  int stop = size;
  if (stop+2 > m_linearized_data_size) { //+2 is for <bos> and <eos>
    stop = m_linearized_data_size-2;
    if (m_verbose && count < 20) {
      ++count;
      LBANN_WARNING("data_id: ", data_id, " smiles string size is ", size, "; losing ", (size-(m_linearized_data_size-2)), " characters");
    }
  }

  data.clear();
  data.reserve(stop+2);
  data.push_back(m_bos);
  for (int j=0; j<stop; j++) {
    const char &w = smiles[j];
    if (m_vocab.find(w) == m_vocab.end()) {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_missing_chars.insert(w);
        ++m_missing_char_in_vocab_count;
        if (m_verbose && m_missing_char_in_vocab_count < 20) {
          std::stringstream ss;
          ss << "world rank: " << m_comm->get_rank_in_world() << "; character not in vocab >>" << w << "<<; idx: " << j << "; data_id: " << data_id << "; string length: " << size << "; will use length: " << stop << "; vocab size: " << m_vocab.size() << std::endl;
          std::cerr << ss.str();
        }
      }
      data.push_back(m_unk);
    } else {
      data.push_back(m_vocab[w]);
    }
  }
  data.push_back(m_eos);
}

void smiles_data_reader::get_sample(int sample_id, std::vector<short> &sample_out) {
  const std::pair<size_t,int>> &p = m_sample_lookup_map(sample_id);
  const size_t &offset = m_sample_lookup_map(sample_id).first;
  int sample_length = m_sample_lookup_map(sample_id).second;
  m_data_fp.seekg(offset);
  sample_out.resize(sample_length);
  int r = fread(raw_data.data(), 1, sample_length);
  if (r != sample_length) {
    LBANN_ERROR("r=", r, "; sample_length=", sample_length, "; should be equal!");
  }
  encode_smiles(raw_data.data(), sample_length, sample_out, sample_id);

}

int smiles_data_reader::get_smiles_string_length(const std::string &line, int line_number) {
  if (m_delimiter == '\0') {
    return line.size();
  }
  size_t k = line.find(m_delimiter);
  if (k == std::string::npos) {
    LBANN_ERROR("failed to find delimit character; as an int: ", (int)m_delimiter, "; line: ", line, " which is line number ", line_number);
  }
  return k;
}

void smiles_data_reader::decode_smiles(const std::vector<short> &data, std::string &out) {
  std::stringstream s;
  for (const auto &t : data) {
    if (!(t == m_eos || t == m_bos || t == m_pad || t == m_unk)) {
      if (m_vocab_inv.find(t) == m_vocab_inv.end()) {
      std::stringstream s2;
      s2 <<"failed to find: " << t <<" in m_vocab_inv for input data: ";
      for (auto tt : data) {
        s2 << tt << " ";
      }
      s2 << "; m_vocab_inv.size(): " << m_vocab_inv.size()
         << " m_vocab_inv keys: ";
      for (auto tt : m_vocab_inv) {
        s2 << tt.first << " ";
      }
      LBANN_ERROR(s2.str());
    }
    }
    const std::string &x = m_vocab_inv[t];
    if (x == "<unk>") {
      s << "<unk>";
    } else if (!(x == "<bos>" || x == "<eos>" || x == "<pad>")) {
      s << m_vocab_inv[t];
    }
  }
  out = s.str();
}

size_t smiles_data_reader::get_mem_usage() const {
  if (m_data_store == nullptr) {
    return m_data.size();
  }
  return m_data_store->get_mem_usage();
}

void smiles_data_reader::get_delimiter() {
  // Get delimiter; default is none ('\0'), though it's likely
  // to be ',' or '\t', since we're likely reading csv files
  options *opts = options::get();
  if (opts->has_string("delimiter")) {
    const std::string d = options::get()->get_string("delimiter");
    const char dd = d[0];
    switch (dd) {
      case 'c' :
        m_delimiter = ',';
        break;
      case 't' :
        m_delimiter = '\t';
        break;
      case '0' :
        m_delimiter = '\0';
        break;
      default :
        LBANN_ERROR("Invalid delimiter character; should be 'c', 't', '0'; you passed: ", d);
    }
  }
  if (is_master()) {
    std::cout << "USING delimiter character: (int)" << (int)m_delimiter << std::endl;
  }
}

void smiles_data_reader::build_sample_lookup_map() {
  double tm1 = get_time();

  // Open input file
  const std::string fn = get_file_dir() + "/" + get_data_filename();
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open data file: ", fn, " for reading");
  }
  std::cout << "opened " << fn << " for reading\n";

  // Loop over lines in the file and construct the lookup map
  std::string line;
  getline(in, line); //discard header
  int sz_prime = m_linearized_data_size-2; // adjust for too long sequences
  while(true) {
    size_t offset = in.tellg();
    getline(in, line);
    if (!line.size()) {
      break; //only exit from loop!
    }
    int sz = get_smiles_string_length(line, data_id);
    if (sz > sz_prime) sz = sz_prime; // adjust for too long sequences
    m_sample_lookup_map.push_back( make_pair(offset, sz) );
  }
  in.close();

  std::cout << "smiles_data_reader::build_sample_lookup_map time: "
            << get_time()-tm1 << std::endl;

  // Open file for reading the actual samples; this is used during fetch_datum
  m_data_fp = fopen(fn.c_str(), "r");
  if (m_data_fp == NULL) {
    LBANN_ERROR("failed to open ", fn, " for reading");
  }
}

}  // namespace lbann
