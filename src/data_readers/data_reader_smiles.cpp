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
}

void smiles_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();

  if (opts->get_bool("ltfb")) {
    opts->set_option("use_data_store", 1);
    opts->set_option("preload_data_store", 1);
  }

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

  // get values for ad-hoc sanity checking; see: do_preload()
  m_min_index = INT_MAX;
  m_max_index = 0;
  for (const auto &idx : m_shuffled_indices) {
    if (idx < m_min_index) m_min_index = idx;
    if (idx > m_max_index) m_max_index = idx;
  }

  get_delimiter();

  // TODO: does this work if we carve off a validation set?
  // NOTE: if ltfb is run, we've hard-coded above to use the
  //       data-store
  if (m_data_store == nullptr) {
    double tm4 = get_time();
    setup_local_cache();
    if (is_master()) {
      std::cout << "time for setup_local_cache(): " << get_time()-tm4
                << "; num samples: " << m_sample_lookup.size() << std::endl;
    }
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
  //m_data_store->set_is_local_cache();
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
  /*for (const auto &id : m_shuffled_indices) {
    valid_ids.insert(id);
    if (id < sanity_min) sanity_min = id;
    if (id > sanity_max) sanity_max = id;
  }*/
  for (size_t idx=0; idx<m_shuffled_indices.size(); idx++) {
    int id = m_shuffled_indices[idx];
    if (id < sanity_min) sanity_min = id;
    if (id > sanity_max) sanity_max = id;
    if (m_data_store->get_index_owner(id) != m_comm->get_rank_in_trainer()) {
      continue;
    }
    valid_ids.insert(id);
  }
  int max_index = sanity_max;

  // cheap sanity check
  if ( (sanity_min != m_min_index || sanity_max != m_max_index)
        &&
        get_role() == "train") {
    LBANN_ERROR("sanity_min != m_min_index || sanity_max != m_max_index: ", sanity_min, " ", m_min_index, " ", sanity_max, " ", m_max_index);
  }

  // Load the samples. As currently written, each rank loads all samples,
  // hence, there will be no data exchange phases. This can be expanded
  // in the future, to use the data store for sharding, instead of a purely
  // local cache
  int sample_id = -1;
  size_t sanity = 0;
  while (true) {
    ++sample_id;
    getline(in, line);
    if (valid_ids.find(sample_id) != valid_ids.end()) {
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
  // no data_store: all data is stored locally
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
    // TODO: change if/when using sharing:
    //   node["/data/" + ...
    const conduit::Node& node = m_data_store->get_conduit_node(data_id);
    const std::string &smiles_string = node["/" + LBANN_DATA_ID_STR(data_id) + "/data"].as_string();
    //const std::string &smiles_string = node["/data/" + LBANN_DATA_ID_STR(data_id) + "/data"].as_string();
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
  encode_smiles(v, size, sample_out, sample_id);
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

// TODO: break into several function calls ??
// TODO: some/most of the following could/should be in the data_store ??
void smiles_data_reader::setup_local_cache() {
  double tm3 = get_time();
  if (is_master()) {
    std::cout << "\nSTARTING smiles_data_reader::setup_fast_experimental() " << std::endl << std::endl;
  }

  // This will hold: (dataum_id, datum_offset, datum length) for each sample
  std::vector<size_t> sample_offsets(m_shuffled_indices.size()*3);

  // Will hold size of above buffer, for bcasting
  size_t buffer_size;

  if (is_master()) {
    double tm1 = get_time();

    // Open input file and discard header line, if it exists
    const std::string infile = get_file_dir() + "/" + get_data_filename();
    std::ifstream in(infile.c_str());
    if (!in) {
      LBANN_ERROR("failed to open data file: ", infile, " for reading");
    }
    std::string line;
    if (m_has_header) {
      getline(in, line);
    }

    // Part 1: compute memory requirements for local cache

    // Get max sample id, which will be the number of lines we need to
    // read from file. This is needed if (1) not using 100% of data,
    // and/or (2) carving off part of train data to use as validation.
    std::unordered_set<int> samples_to_use;
    int max_sample_id = 0;
    for (size_t j=0; j<m_shuffled_indices.size(); j++) {
      samples_to_use.insert(m_shuffled_indices[j]);
      max_sample_id = m_shuffled_indices[j] > max_sample_id ? m_shuffled_indices[j] : max_sample_id;
    }
    ++max_sample_id;

    // Construct sample_offsets vector
    sample_offsets.clear();
    size_t offset = 0;
    for (int j=0; j<max_sample_id; j++) {
      getline(in, line);
      if (line.size() < 5) {
        LBANN_ERROR("read ", j, " lines from file; could not read another. --num_samples is probably incorrect");
      }
      if (samples_to_use.find(j) != samples_to_use.end()) {
        int k = get_smiles_string_length(line, j);
        sample_offsets.push_back(j);
        sample_offsets.push_back(offset);
        sample_offsets.push_back(k);
        offset += k;
      }
    }
    buffer_size = offset;
    m_data.resize(buffer_size);

    // Part 2: Fill in the data buffer
    in.seekg(0);
    if (m_has_header) {
      getline(in, line);
    }
    offset = 0;
    for (int j=0; j<max_sample_id; j++) {
      getline(in, line);
      if (samples_to_use.find(j) != samples_to_use.end()) {
        int k = get_smiles_string_length(line, j);
        for (int n=0; n<k; n++) {
          m_data[n+offset] = line[n];
        }
        offset += k;
      }
    }

    if (sample_offsets.size()/3 != m_shuffled_indices.size()) {
      LBANN_ERROR("sample_offsets.size()/3: ", sample_offsets.size()/3, " should be equal to m_shuffled_indices.size which is ", m_shuffled_indices.size());
    }
    std::cout << "P_0 time for computing sample sizes and filling buffer: " << get_time() - tm1 << std::endl;
  }

  // Construct lookup table for locating samples in the m_data vector (aka, the sample buffer)
  m_comm->broadcast<size_t>(0, sample_offsets.data(), sample_offsets.size(), m_comm->get_world_comm());
  for (size_t j=0; j<sample_offsets.size(); j += 3) {
    m_sample_lookup[sample_offsets[j]] =
      std::make_pair(sample_offsets[j+1], sample_offsets[j+2]);
  }

  // Bcast the sample buffer
  m_comm->broadcast<size_t>(0, &buffer_size, 1, m_comm->get_world_comm());
  m_data.resize(buffer_size);

  int full_rounds = m_data.size() / INT_MAX;
  int last_round = m_data.size() % INT_MAX;
  size_t the_offset = 0;

  for (int j=0; j<full_rounds; j++) {
    m_comm->broadcast<char>(0, m_data.data()+the_offset, INT_MAX, m_comm->get_world_comm());
    the_offset += INT_MAX;
  }
  if (last_round) {
    m_comm->broadcast<char>(0, m_data.data()+the_offset, last_round, m_comm->get_world_comm());
  }

  if (is_master()) {
    std::cout << "total time for loading data: " << get_time()-tm3 << std::endl
              << "num samples: " << m_sample_lookup.size() << std::endl;
  }

  // Only used for testing/debugging during development
  if (options::get()->get_bool("test_encode")) {
    test_encode();
  }

  if (is_master()) {
    pid_t p = getpid();
    char buf[80];
    sprintf(buf, "cat /proc/%d/status >& status.txt", p);
    system(buf);
    std::cout << "wrote proc/[pid]/status to 'status.txt'" << std::endl;
  }
}

void smiles_data_reader::test_encode() {
  // What this does: at this point, P_0 has read and bcast the data set,
  // and each rank has built a lookup table. Below, P_1 looks up each
  // data_id; encodes the string (E1); reads the string from file (S2);
  // decodes E1 to produce string S1; compares S1 and S2 for equality.
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "STARTING TEST_ENCODE" << std::endl;
  }
  if (m_comm->get_rank_in_world() != 1) {
    return;
  }

  // option: testing the test ;)
  bool fail = options::get()->get_bool("make_test_fail");

  // Build ordered set of data_ids so we can more easily iterate
  // through the file -- instead of jumping around
  std::set<int> data_ids;
  for (auto t : m_sample_lookup) {
    data_ids.insert(t.first);
  }

  // Open input file and discard header (if it exists)
  const std::string infile = get_file_dir() + "/" + get_data_filename();
  std::ifstream in(infile.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", infile, " for reading");
  }
  std::string line;
  if (m_has_header) {
    getline(in, line);
  }

  size_t num_tested = 0;

  std::vector<short> encoded;
  std::string decoded;
  int sample_id = -1;
  while (getline(in, line)) {
    ++sample_id;

    if (data_ids.find(sample_id) != data_ids.end()) {
      ++num_tested;
      // encode then decode the datum that is stored in memory
      get_sample(sample_id, encoded);
      decode_smiles(encoded, decoded);

      // get datum length from the line we've just read from file
      size_t k = get_smiles_string_length(line, sample_id);
      std::string S2(line.data(), k);

      // test the test! Optionally make the test fail;
      // assumes smiles string contains at least 8 characters,
      // and no string contains "~~~~"
      if (num_tested > 10 && fail) {
        for (size_t h=0; h<S2.size(); h++) {
          S2[h] = '~';
        }
      }

      // conduct tests
      // It would be simpler to throw exceptions here, but currently that
      // would cause all other procs to hang
      if (S2.size() != decoded.size()) {
        LBANN_ERROR("S2.size (", S2.size(), ") != decoded.size (", decoded.size());
      }
      if (S2 != decoded) {
        LBANN_ERROR("test_encoded failed; string from memory: ", decoded, "; string from file: ", S2, "; should be equal");
      }
    }
  }
  in.close();

  if (num_tested != m_sample_lookup.size()) {
    LBANN_ERROR("num_tested= ", num_tested, "; m_sample_lookup.size()= ", m_sample_lookup.size(), "; should be equal");
  }

  std::cout << "ENDING TEST_ENCODE; time: " << get_time()-tm1
            << " >>> TESTS PASSED <<< " << std::endl;
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

}  // namespace lbann
