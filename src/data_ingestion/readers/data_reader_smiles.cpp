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

#include "lbann/data_ingestion/readers/data_reader_smiles.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_ingestion/data_store_conduit.hpp"
#include "lbann/data_ingestion/readers/data_reader_sample_list_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list_open_files_impl.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/vectorwrapbuf.hpp"
#include <algorithm>
#include <cctype>
#include <mutex>
#include <random>

namespace lbann {

smiles_data_reader::smiles_data_reader(const bool shuffle)
  : data_reader_sample_list(shuffle)
{}

smiles_data_reader::smiles_data_reader(const smiles_data_reader& rhs)
  : data_reader_sample_list(rhs)
{
  copy_members(rhs);
}

smiles_data_reader::~smiles_data_reader() {}

smiles_data_reader& smiles_data_reader::operator=(const smiles_data_reader& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  data_reader_sample_list::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void smiles_data_reader::copy_members(const smiles_data_reader& rhs)
{
  data_reader_sample_list::copy_members(rhs);
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
  m_metadata_filename = rhs.m_metadata_filename;
  m_missing_char_in_vocab_count = rhs.m_missing_char_in_vocab_count;
  m_missing_chars = rhs.m_missing_chars;
  m_vocab = rhs.m_vocab;
  m_vocab_inv = rhs.m_vocab_inv;

  m_index_to_local_id = rhs.m_index_to_local_id;
  m_local_to_index = rhs.m_local_to_index;
  m_filename_to_local_id_set = rhs.m_filename_to_local_id_set;
  m_index_to_filename = rhs.m_index_to_filename;
}

void smiles_data_reader::load()
{
  if (get_comm()->am_world_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  double tm1 = get_time();
  auto& arg_parser = global_argument_parser();

  // for now, only implemented for data store with preloading
  set_use_data_store(true);

  if (m_sequence_length == 0) {
    if (arg_parser.get<int>(LBANN_OPTION_SEQUENCE_LENGTH) == -1) {
      LBANN_ERROR("you must pass --sequence_length=<int> on the cmd line or "
                  "call set_sequence_length()");
    }
    m_sequence_length = arg_parser.get<int>(LBANN_OPTION_SEQUENCE_LENGTH);
  }
  m_linearized_data_size = m_sequence_length + 2;

  // load the vocabulary; this is a map: string -> short
  if (m_vocab.size() == 0) {
    if (arg_parser.get<std::string>(LBANN_OPTION_VOCAB) == "") {
      LBANN_ERROR("you must either pass --vocab=<string> on the command line "
                  "or call load_vocab(...)");
    }
    const std::string fn = arg_parser.get<std::string>(LBANN_OPTION_VOCAB);
    load_vocab(fn);
  }
  else {
    LBANN_ERROR("you passed --vocab=<string>, but it looks like load_vocab() "
                "was previously called. You must use one or the other.");
  }

  // Load the sample list(s)
  data_reader_sample_list::load();
  if (get_comm()->am_world_master()) {
    std::cout << "time to load sample list: " << get_time() - tm1 << std::endl;
  }

  // do what we almost always do (TODO: this should be refactored as a method
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();

  // load various metadata
  build_some_maps();
  load_offsets_and_lengths();
  print_statistics();
}

void smiles_data_reader::do_preload_data_store()
{
  double tm1 = get_time();
  if (get_comm()->am_world_master()) {
    std::cout << "starting do_preload_data_store; num indices: "
              << utils::commify(m_shuffled_indices.size())
              << "; role: " << get_role() << std::endl;
  }

  // Randomize the ordering in which this rank will open data files.
  // Note that each rank will open each data file at most one time.
  std::vector<std::string> my_ordering;
  my_ordering.reserve(m_local_to_index.size());
  for (std::unordered_map<std::string, std::map<size_t, size_t>>::const_iterator
         fn_iter = m_local_to_index.begin();
       fn_iter != m_local_to_index.end();
       ++fn_iter) {
    const std::string filename = fn_iter->first;
    my_ordering.push_back(filename);
  }
  std::random_device rd;
  std::mt19937 r(rd());
  std::shuffle(my_ordering.begin(), my_ordering.end(), r);

  auto& arg_parser = global_argument_parser();
  size_t buf_size = arg_parser.get<size_t>(LBANN_OPTION_SMILES_BUFFER_SIZE);
  // Create a buffer for reading in the SMILES files
  std::vector<char> iobuffer(buf_size);

  // load all samples that belong to this rank's data_store
  for (const auto& filename : my_ordering) {
    std::ifstream in;
    in.rdbuf()->pubsetbuf(iobuffer.data(), buf_size);
    in.open(filename.c_str(), std::ios::binary | std::ios::ate);
    if (!in) {
      LBANN_ERROR("failed to open ", filename, " for reading");
    }

    const std::map<size_t, size_t>& local_to_index = m_local_to_index[filename];
    size_t min_offset = std::numeric_limits<size_t>::max();
    size_t max_offset = 0;
    size_t len_of_last_offset = 0;

    // Create local batches for fetching from a contiguous buffer
    std::map<size_t, size_t> samples_in_range;

    std::vector<std::pair<size_t, size_t>> loadme;
    loadme.reserve(local_to_index.size());
    for (auto const& [local, index] : local_to_index) {
      loadme.push_back(std::make_pair(local, index));
    }

    for (size_t k = 0; k < loadme.size(); k++) {
      size_t local = loadme[k].first;
      size_t index = loadme[k].second;
      offset_map_t::const_iterator iter = m_sample_offsets.find(index);
      if (iter == m_sample_offsets.end()) {
        LBANN_ERROR("failed to find ",
                    index,
                    " in m_sample_offsets map; map size: ",
                    m_sample_offsets.size());
      }

      samples_in_range[local] = index;

      // Check to see if the current sample can be fetched as part
      // of a range based fetch
      const offset_t& d = iter->second;
      const size_t offset = d.first;
      if (offset < min_offset) {
        min_offset = offset;
      }

      // dah - 'length' is length of the SMILES string, not necessarily
      //       line length, which will be longer for non-trivial delimiters
      size_t line_length = d.second;
      if (k + 1 < loadme.size()) {
        size_t next_index = loadme[k + 1].second;
        offset_map_t::const_iterator next_iter =
          m_sample_offsets.find(next_index);
        if (next_iter == m_sample_offsets.end()) {
          LBANN_ERROR("failed to find ",
                      index,
                      " in m_sample_offsets map; map size: ",
                      m_sample_offsets.size());
        }
        const offset_t& next_d = next_iter->second;
        const size_t next_offset = next_d.first;
        line_length = next_offset - offset;
      }

      // dah - "if len_of_last_offset == 0" deals with the edge case:
      //       only one sample in the file
      //       (should we check for empty file?)
      if (offset > max_offset || len_of_last_offset == 0) {
        max_offset = offset;
        len_of_last_offset = line_length;
      }

      // Once enough samples have been identified in a range or the
      // end of the map is reached:
      // fetch a chunk of the buffer and load the samples
      // Read in at least one extra character for validating the end
      // of the string
      size_t read_len =
        max_offset + len_of_last_offset - min_offset + sizeof(char);
      if ((read_len >= (buf_size - (m_linearized_data_size + sizeof(char)))) ||
          local == local_to_index.rbegin()->first) {
        // Read a large chunk from the file
        in.seekg(min_offset, std::ios::beg);
        std::vector<char> buffer(read_len);
        in.read(buffer.data(), read_len);
        std::istringstream buf_stream({buffer.data(), buffer.size()});

        // Place all of the samples in the range into the data store
        for (const auto& [r_local, r_index] : samples_in_range) {
          (void)r_local; // silence compiler warning about unused variable.
          // BVE CHECK THIS
          if (m_data_store->get_index_owner(r_index) !=
              get_comm()->get_rank_in_trainer()) {
            continue;
          }

          // build conduit node
          conduit::Node& node = m_data_store->get_empty_node(r_index);
          construct_conduit_node(node, &buf_stream, r_index, min_offset);
          m_data_store->set_preloaded_conduit_node(r_index, node);
        }

        // The range is complete, reset
        min_offset = std::numeric_limits<size_t>::max();
        max_offset = 0;
        len_of_last_offset = 0;
        samples_in_range.clear();
      }
    } // for (auto const& [local, index] : local_to_index)
    in.close();
  }

  if (get_comm()->am_world_master()) {
    std::cout << " do_preload_data_store time: " << get_time() - tm1
              << std::endl;
  }
}

std::set<int> smiles_data_reader::get_my_indices() const
{
  std::set<int> s;
  for (size_t j = 0; j < m_shuffled_indices.size(); j++) {
    const int index = m_shuffled_indices[j];
    if (m_data_store->get_index_owner(index) ==
        get_comm()->get_rank_in_trainer()) {
      s.insert(index);
    }
  }
  return s;
}

bool smiles_data_reader::fetch_datum(Mat& X, uint64_t data_id, uint64_t mb_idx)
{
  if (!data_store_active()) {
    LBANN_ERROR(
      "it should be impossible you you to be here; please contact Dave Hysom");
  }

  const conduit::Node& node = m_data_store->get_conduit_node(data_id);
  const conduit::unsigned_short_array data =
    node["/" + LBANN_DATA_ID_STR(data_id) + "/data"].as_unsigned_short_array();
  size_t j;
  size_t n = data.number_of_elements();
  for (j = 0; j < n; ++j) {
    X(j, mb_idx) = data[j];
  }
  for (; j < static_cast<size_t>(m_linearized_data_size); j++) {
    X(j, mb_idx) = m_pad;
  }
  return true;
}

bool smiles_data_reader::fetch_label(Mat& Y, uint64_t data_id, uint64_t mb_idx)
{
  LBANN_ERROR("smiles_data_reader::fetch_label is not implemented");
  return true;
}

bool smiles_data_reader::fetch_response(Mat& Y,
                                        uint64_t data_id,
                                        uint64_t mb_idx)
{
  LBANN_ERROR("smiles_data_reader::fetch_response is not implemented");
  return true;
}

// user feedback
void smiles_data_reader::print_statistics() const
{
  if (!get_comm()->am_world_master()) {
    return;
  }

  std::cerr << "\n======================================================\n";
  std::cerr << "role: " << get_role() << std::endl;
  // std::cerr << "mem for data, lower bound: " <<
  // utils::commify(get_mem_usage()) << std::endl;
  std::cerr << "num samples per trainer: "
            << utils::commify(m_shuffled_indices.size()) << std::endl;
  std::cerr << "max sequence length: " << utils::commify(m_linearized_data_size)
            << std::endl;
  std::cerr << "num features=" << utils::commify(m_linearized_data_size)
            << std::endl;
  std::cerr << "pad index: " << m_pad << std::endl;

  if (m_missing_chars.size()) {
    std::cerr << std::endl
              << "The following tokens were in SMILES strings, but were "
                 "missing from the vocabulary: ";
    for (const auto t : m_missing_chars) {
      std::cerr << t << " ";
    }
    std::cerr << std::endl;
  }

  // +4 for <bos>, <eos>, <unk>, <pad>
  std::cerr << "vocab size: " << m_vocab.size() + 4 << std::endl
            << "    (includes +4 for <bos>, <eos>, <pad>, <unk>)" << std::endl
            << "======================================================\n\n";
}

void smiles_data_reader::load_vocab(std::string fn)
{
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ",
                fn,
                " for reading; this is the vocabulary file");
  }
  std::stringstream s;
  s << in.rdbuf();
  in.close();
  load_vocab(s);
}

void smiles_data_reader::load_vocab(std::stringstream& in)
{
  // TODO: trainer master should read and bcast
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
  if (sanity) {
    LBANN_ERROR("failed to find <pad> and/or <unk> and/or <bos> and/or <eos> "
                "in vocab input stream");
  }
}

bool smiles_data_reader::encode_smiles(const std::string& smiles,
                                       std::vector<unsigned short>& data)
{
  return encode_smiles(smiles.data(), smiles.size(), data);
}

bool smiles_data_reader::encode_smiles(const char* smiles,
                                       unsigned short size,
                                       std::vector<unsigned short>& data)
{
  static int count = 0;
  bool found_all_characters_in_vocab = true;

  int stop = size;
  if (stop + 2 > m_linearized_data_size) { //+2 is for <bos> and <eos>
    stop = m_linearized_data_size - 2;
    if (m_verbose && count < 20) {
      ++count;
      LBANN_WARNING("smiles string size is ",
                    size,
                    "; losing ",
                    (size - (m_linearized_data_size - 2)),
                    " characters; m_sequence_length: ",
                    m_linearized_data_size);
    }
  }

  data.clear();
  data.reserve(stop + 2);
  data.push_back(m_bos);
  for (int j = 0; j < stop; j++) {
    const char& w = smiles[j];
    if (m_vocab.find(w) == m_vocab.end()) {
      found_all_characters_in_vocab = false;
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_missing_chars.insert(w);
        ++m_missing_char_in_vocab_count;
        if (m_verbose && m_missing_char_in_vocab_count < 20 &&
            m_comm != nullptr) {
          std::stringstream ss;
          ss << "world rank: " << m_comm->get_rank_in_world()
             << "; character not in vocab >>" << w << "<<; idx: " << j
             << "; string length: " << size << "; will use length: " << stop
             << "; vocab size: " << m_vocab.size() << std::endl;
          std::cerr << ss.str();
        }
      }
      data.push_back(m_unk);
    }
    else {
      data.push_back(m_vocab[w]);
    }
  }
  data.push_back(m_eos);

  while (data.size() < static_cast<size_t>(m_linearized_data_size)) {
    data.push_back(m_pad);
  }
  return found_all_characters_in_vocab;
}

void smiles_data_reader::decode_smiles(const std::vector<unsigned short>& data,
                                       std::string& out)
{
  std::stringstream s;
  for (const auto& t : data) {
    if (!(t == m_eos || t == m_bos || t == m_pad || t == m_unk)) {
      if (m_vocab_inv.find(t) == m_vocab_inv.end()) {
        std::stringstream s2;
        s2 << "failed to find: " << t << " in inv_map; here is the data: ";
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
    const std::string& x = m_vocab_inv[t];
    if (x == "<unk>") {
      s << "<unk>";
    }
    else if (!(x == "<bos>" || x == "<eos>" || x == "<pad>")) {
      s << m_vocab_inv[t];
    }
  }
  out = s.str();
}

void smiles_data_reader::load_offsets_and_lengths()
{
  // trainer P_0 fills in offset_data vector, then bcasts
  std::vector<SampleData> offset_data;

  if (get_comm()->am_trainer_master()) {
    read_offset_data(offset_data);
  }
  size_t n_samples = offset_data.size(); // only meaningful for root
  get_comm()->trainer_broadcast<size_t>(0, &n_samples, 1);
  offset_data.resize(n_samples); // not meaningful for root
  get_comm()->trainer_broadcast<SampleData>(0,
                                            offset_data.data(),
                                            offset_data.size());

  // fill in the m_sample_offsets map
  for (size_t j = 0; j < offset_data.size(); j++) {
    const SampleData& d = offset_data[j];
    m_sample_offsets[d.index] = std::make_pair(d.offset, d.length);
  }
}

void smiles_data_reader::construct_conduit_node(conduit::Node& node,
                                                std::istream* istrm,
                                                size_t index,
                                                size_t buf_offset)
{
  std::vector<unsigned short> sample;
  load_sample(istrm, index, sample, buf_offset);
  node[LBANN_DATA_ID_STR(index) + "/data"] = sample;
}

void smiles_data_reader::load_sample(std::istream* istrm,
                                     size_t index,
                                     std::vector<unsigned short>& output,
                                     size_t buf_offset)
{
  const std::string smiles_str = get_raw_sample(istrm, index, buf_offset);
  encode_smiles(smiles_str, output);
}

std::string smiles_data_reader::get_raw_sample(std::istream* istrm,
                                               size_t index,
                                               size_t buf_offset)
{
  offset_map_t::const_iterator iter = m_sample_offsets.find(index);
  if (iter == m_sample_offsets.end()) {
    LBANN_ERROR("failed to find ",
                index,
                " in m_sample_offsets map; map size: ",
                m_sample_offsets.size());
  }
  const offset_t& d = iter->second;
  const long long& offset = d.first;
  const short length = d.second;
  size_t start = offset - buf_offset;
  // check that string is at beginning of line
  if (start) {
    istrm->seekg(offset - buf_offset - 1);
    char c;
    istrm->read((char*)&c, sizeof(char));
    if (c != '\n') {
      std::stringstream s;
      s << "string does not start after a newline; previous char, as int: \n"
        << "  offset: " << offset << " Length: " << length
        << " as int: " << (int)c << " tellg: " << istrm->tellg();
      if (isprint(c)) {
        s << " as char: " << c;
      }
      else {
        s << " not a printable character";
      }
      LBANN_ERROR(s.str());
    }
  }
  else {
    istrm->seekg(offset - buf_offset);
  }

  std::string smiles_str;
  smiles_str.resize(length);
  istrm->read((char*)smiles_str.data(), length);

  // check that next char is a valid delimiter
  int c2 = istrm->peek();
  if (!(istrm->eof() || is_delimiter(c2))) {
    std::stringstream s;
    s << "string does not appear to be followed by a whitespace (or valid "
         "delimiter); "
      << "the string: >>>" << smiles_str << "<<<; next char, as int: " << c2
      << "; tellg: " << istrm->tellg();
    if (isprint(c2)) {
      s << " as char: " << (char)c2;
    }
    else {
      s << " not a printable character";
    }
    LBANN_ERROR(s.str());
  }
  // Check the input string for any internal delimiters and truncate
  // if found
  size_t SMILES_len = smiles_str.length();
  for (std::string::iterator it = smiles_str.begin(); it != smiles_str.end();
       ++it) {
    auto& c = *it;
    if (is_delimiter(c)) {
      SMILES_len = std::distance(smiles_str.begin(), it);
      break;
    }
  }
  return smiles_str.substr(0, SMILES_len);
}

void smiles_data_reader::build_some_maps()
{
  for (const auto& index : m_shuffled_indices) {
    auto const [file_id, local_id] = get_sample(index);
    std::stringstream s;
    std::string dir = m_sample_list.get_samples_dirname();
    s << dir << "/" << m_sample_list.get_samples_filename(file_id);
    std::string filename = s.str();
    // this has bit me before, and can be an easy mistake to make
    // when writing sample lists:
    file::remove_multiple_slashes(filename);

    // construct map entries
    m_index_to_local_id[index] = local_id;
    m_local_to_index[filename][local_id] = index;
    m_index_to_filename[index] = filename;
    m_filename_to_local_id_set[filename].insert(local_id);
  }
}

void smiles_data_reader::read_offset_data(std::vector<SampleData>& data)
{
  data.clear();

  // read the metadata file; each line contains: filename,
  // corresponding offsets_filename, num_samples
  std::vector<size_t> samples_per_file;
  std::vector<std::string> data_filenames;
  std::vector<std::string> offsets_filenames;
  read_metadata_file(samples_per_file, data_filenames, offsets_filenames);

  auto& arg_parser = global_argument_parser();
  size_t buf_size = arg_parser.get<size_t>(LBANN_OPTION_SMILES_BUFFER_SIZE);
  // Create a buffer for reading in each offsets file
  std::vector<char> iobuffer(buf_size);
  long long offset;
  unsigned short length;
  for (size_t j = 0; j < data_filenames.size(); j++) {
    if (m_filename_to_local_id_set.find(data_filenames[j]) !=
        m_filename_to_local_id_set.end()) {
      std::ifstream in;
      in.rdbuf()->pubsetbuf(iobuffer.data(), buf_size);
      in.open(offsets_filenames[j], std::ios::binary);

      const std::set<size_t>& indices =
        m_filename_to_local_id_set[data_filenames[j]];

      if (indices.size() == 0) {
        continue;
      }

      std::set<size_t> indices_in_range;
      size_t min_id = std::numeric_limits<size_t>::max();
      size_t max_id = 0;

      for (const auto& local_id : indices) {
        indices_in_range.insert(local_id);
        if (local_id < min_id) {
          min_id = local_id;
        }
        if (local_id > max_id) {
          max_id = local_id;
        }
        /// If the range of IDs exceeds the buffer fetch size, or we
        /// are at the end of the indices:
        /// Fetch the buffer and read out the offsets
        size_t read_len = (max_id + 1 - min_id) * OffsetAndLengthBinarySize;
        if ((read_len >= (buf_size - OffsetAndLengthBinarySize)) ||
            local_id == *(indices.rbegin())) {

          // Read a large chunk from the file
          size_t start_offset = min_id * OffsetAndLengthBinarySize;
          in.seekg(start_offset, std::ios::beg);
          std::vector<char> buffer(read_len);
          if (!in.read(buffer.data(), read_len)) {
            LBANN_ERROR("read buffer failed : ", offsets_filenames[j]);
          }
          std::istringstream buf_stream({buffer.data(), buffer.size()});

          // Extract all of the indices in the range
          for (const auto& r_local_id : indices_in_range) {
            if (!buf_stream.seekg(r_local_id * OffsetAndLengthBinarySize -
                                  start_offset)) {
              LBANN_ERROR("seek failed");
            }
            if (!buf_stream.read((char*)&offset, OffsetBinarySize)) {
              LBANN_ERROR("read offset field failed");
            }
            if (!buf_stream.read((char*)&length, LengthBinarySize)) {
              LBANN_ERROR("read length field failed");
            }
            size_t index = m_local_to_index[data_filenames[j]][r_local_id];
            data.push_back(SampleData(index, offset, length));
          }

          // The range is complete, reset
          min_id = std::numeric_limits<size_t>::max();
          max_id = 0;
          indices_in_range.clear();
        }
      }
      in.close();
    }
  }
}

void smiles_data_reader::read_metadata_file(
  std::vector<size_t>& samples_per_file,
  std::vector<std::string>& data_filenames,
  std::vector<std::string>& offsets_filenames)
{
  // open the metadata file
  const std::string metadata_fn = get_metadata_filename();
  if (metadata_fn.empty()) {
    LBANN_ERROR("label filename is empty");
  }
  std::ifstream istrm(metadata_fn.c_str());
  if (!istrm) {
    LBANN_ERROR("failed to open ", metadata_fn, " for reading");
  }

  // clear output variables
  samples_per_file.clear();
  data_filenames.clear();
  offsets_filenames.clear();

  std::string data_filename;
  std::string offsets_filename;
  size_t n_samples;
  std::string line;
  while (std::getline(istrm, line)) {
    // skip comment and empty lines (3 is a magic number)
    if (line.size() < 3 || line[0] == '#') {
      continue;
    }
    std::stringstream s(line);
    s >> n_samples >> data_filename >> offsets_filename;

    data_filenames.push_back(data_filename);
    offsets_filenames.push_back(offsets_filename);
    samples_per_file.push_back(n_samples);
  }
  istrm.close();

  // ======= sanity checks =======
  for (size_t j = 0; j < data_filenames.size(); j++) {
    n_samples = samples_per_file[j];
    std::ifstream in(offsets_filenames[j].c_str(), std::ios::binary);
    if (!in) {
      LBANN_ERROR("failed to open ", offsets_filenames[j], " for reading");
    }
    in.seekg(0, std::ios::end);
    size_t bytes = in.tellg();
    in.close();
    size_t num_samples_cur_file = bytes / OffsetAndLengthBinarySize;
    // Quick test: does the number of state samples match the count from the
    // offsets file? This is mostly to detect off-by-one, or other, errors in
    // the sample list
    if (num_samples_cur_file != n_samples) {
      LBANN_ERROR("total num samples in ",
                  data_filenames[j],
                  " from offsets file is ",
                  num_samples_cur_file,
                  " but metadata file claimed the number is ",
                  n_samples);
    }
    // if (m_filename_to_local_id_set.find(data_filename) ==
    // m_filename_to_local_id_set.end()) {
    //   LBANN_WARNING("Unused metadata mapping: failed to find ",
    //   data_filename, " in m_filename_to_local_id_set; ", " num keys: ",
    //   m_filename_to_local_id_set.size());
    // }
  }
  // END ======= sanity checks =======
}

void smiles_data_reader::use_unused_index_set(execution_mode m)
{
  data_reader_sample_list::use_unused_index_set(m);
  // Clear the existing data structures
  m_index_to_local_id.clear();
  m_local_to_index.clear();
  m_filename_to_local_id_set.clear();
  // Rebuild them on the previously used index set
  build_some_maps();
  load_offsets_and_lengths();
  print_statistics();
}

void smiles_data_reader::get_sample_origin(const size_t index_in,
                                           std::string& filename_out,
                                           size_t& offset_out,
                                           unsigned short& length_out) const
{

  offset_map_t::const_iterator t2 = m_sample_offsets.find(index_in);
  if (t2 == m_sample_offsets.end()) {
    LBANN_ERROR("index ", index_in, " not found in m_sample_offsets");
  }
  std::unordered_map<size_t, std::string>::const_iterator t3 =
    m_index_to_filename.find(index_in);
  if (t3 == m_index_to_filename.end()) {
    LBANN_ERROR("index ", index_in, " not found in  m_index_to_filename");
  }

  offset_out = t2->second.first;
  length_out = t2->second.second;
  filename_out = t3->second;
}

void smiles_data_reader::set_offset(size_t index,
                                    long long offset,
                                    unsigned short length)
{
  m_sample_offsets[index] = std::make_pair(offset, length);
}

} // namespace lbann
