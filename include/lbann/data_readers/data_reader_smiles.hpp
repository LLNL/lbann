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

#ifndef LBANN_DATA_READER_SMILES_HPP
#define LBANN_DATA_READER_SMILES_HPP

#include "lbann/data_readers/data_reader_sample_list.hpp"
#include "lbann/data_readers/sample_list_ifstream.hpp"

namespace lbann {
/**
 * Data reader for SMILES (string) data. The string data is converted to
 * a vector of shorts according to an arbitrary mapping.
 *
 * Terminology and Notes:
 *   "local_id" (or similar name): refers to a line number in a file.
 *   "global_id" (aka, sample_id, etc) refers to an index from the
 *               m_shuffled_indices vector.
 */
class smiles_data_reader
  : public data_reader_sample_list<sample_list_ifstream<long long>>
{
public:
  // Types for mapping a sample id to an <offset,length> locator
  using offset_t = std::pair<long long, unsigned short>;
  using offset_map_t = std::unordered_map<size_t, offset_t>;

  smiles_data_reader(const bool shuffle);
  smiles_data_reader(const smiles_data_reader&);
  smiles_data_reader& operator=(const smiles_data_reader&);
  ~smiles_data_reader() override;

  smiles_data_reader* copy() const override
  {
    return new smiles_data_reader(*this);
  }

  std::string get_type() const override { return "smiles_data_reader"; }

  void load() override;

  int get_linearized_data_size() const override
  {
    return m_linearized_data_size;
  }
  int get_linearized_label_size() const override
  {
    return m_linearized_label_size;
  }
  int get_linearized_response_size() const override
  {
    return m_linearized_response_size;
  }
  const std::vector<El::Int> get_data_dims() const override
  {
    return {get_linearized_data_size()};
  }
  int get_num_labels() const override { return m_num_labels; }

  void set_sequence_length(int n)
  {
    m_sequence_length = n;
    m_linearized_data_size = n + 2;
  }
  int get_sequence_length() { return m_sequence_length; }

  void use_unused_index_set(execution_mode m) override;

  /** This method is for use during testing and development */
  void get_sample_origin(const size_t index_in,
                         std::string& filename_out,
                         size_t& offset_out,
                         unsigned short& length_out) const;

  /** This method is for use during testing and development.
   *  Returns the set of indices whose samples are cached in
   *  the data_store
   */
  std::set<int> get_my_indices() const;

  /** This method made public for use during testing.
   *  Convert SMILES string to a vector of shorts
   */
  bool encode_smiles(const char* smiles,
                     unsigned short size,
                     std::vector<unsigned short>& data);
  /** This method made public for use during testing.
   *  Convert SMILES string to a vector of shorts
   */
  bool encode_smiles(const std::string& smiles,
                     std::vector<unsigned short>& data);
  /** This method made public for use during testing.
   *  Decode SMILES string from a vector of shorts
   */
  void decode_smiles(const std::vector<unsigned short>& data, std::string& out);

  /** This method made public for use during testing. */
  void load_vocab(std::string filename);
  /** This method made public for use during testing. */
  void load_vocab(std::stringstream& s);
  /** This method made public for use during testing. */
  void set_linearized_data_size(size_t s) { m_linearized_data_size = s; }
  /** This method made public for use during testing. */
  // reads and returns the smiles string from the input stream
  std::string
  get_raw_sample(std::istream* istrm, size_t index, size_t buf_offset = 0);
  /** This method made public only for use during testing.
   *  Insert an entry into a map: index -> (offset, length),
   *  where 'index' is an alias for an entry in the shuffled_indices
   */
  void set_offset(size_t index, long long offset, unsigned short length);

  /** This method made public for use during testing. */
  void load_list_of_samples(const std::string sample_list_file);

  /** @brief Sets the name of the metadata file */
  void set_metadata_filename(std::string fn)
  {
    m_metadata_filename = std::move(fn);
  }

  /** @brief Returns the name of the metadata file */
  const std::string& get_metadata_filename() { return m_metadata_filename; }

private:
  // note: linearized_size is m_sequence_length+2; the +2 is for the
  //       <bos> and <eos> characters that get tacked on
  int m_sequence_length = 0;

  const size_t OffsetBinarySize = sizeof(long long);
  const size_t LengthBinarySize = sizeof(unsigned short);
  const std::streamsize OffsetAndLengthBinarySize =
    OffsetBinarySize + LengthBinarySize;

  struct SampleData
  {
    SampleData() {}
    SampleData(int idx, long long off, unsigned short len)
      : index(idx), offset(off), length(len)
    {}
    size_t index;
    long long offset;
    unsigned short length;
  };

  int m_linearized_data_size = 0;
  int m_linearized_label_size = 0;
  int m_linearized_response_size = 0;
  int m_num_labels = 0;

  // these may be changed when the vocab file is read
  short m_pad = 420;
  short m_unk = 421;
  short m_bos = 422;
  short m_eos = 423;

  std::string m_metadata_filename;

  std::unordered_map<char, short> m_vocab;
  std::unordered_map<short, std::string> m_vocab_inv;

  std::mutex m_mutex;

  size_t m_missing_char_in_vocab_count = 0;
  std::unordered_set<char> m_missing_chars;

  // maps: sample id -> offset within a file
  offset_map_t m_sample_offsets;

  /** Here and elsewhere, 'index' refers to an entry in the shuffled indices;
   *  'local_id' refers, loosely, to a line number in a file.
   *  Also, 'file' or 'filename' refers to a file containing SMILES strings.
   *
   *  maps: index -> local_id. Note: clearly the mapping is not unique,
   *  since a local_id requires a filename for full specification
   */
  std::unordered_map<size_t, size_t> m_index_to_local_id;

  /** maps: filename -> { local_id -> index } */
  std::unordered_map<std::string, std::map<size_t, size_t>> m_local_to_index;

  /** maps: filename -> {local_id} */
  std::unordered_map<std::string, std::set<size_t>> m_filename_to_local_id_set;

  /** maps: index -> filename */
  std::unordered_map<size_t, std::string> m_index_to_filename;

  //=====================================================================
  // private methods follow
  //=====================================================================

  /** @brief Contains common code for operator= and copy ctor */
  void copy_members(const smiles_data_reader& rhs);

  void do_preload_data_store() override;

  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;

  void print_statistics() const;

  // load "offset" and "length" for samples from a binary file;
  // the (offset, length) specify the location of a sample within
  // the data file
  void load_offsets_and_lengths();

  // called by load_offsets_and_lengths
  void read_offset_data(std::vector<SampleData>& data);

  // calls load_sample
  void construct_conduit_node(conduit::Node& node,
                              std::istream* istream,
                              size_t sample_id,
                              size_t buf_offset = 0);

  // calls get_raw_sample; returns in 'output' an encoded version of the sample
  void load_sample(std::istream* istrm,
                   size_t index,
                   std::vector<unsigned short>& output,
                   size_t buf_offset = 0);

  void build_some_maps();

  // called by read_offset_data()
  void read_metadata_file(std::vector<size_t>& samples_per_file,
                          std::vector<std::string>& data_filenames,
                          std::vector<std::string>& offsets_filenames);

  bool is_delimiter(const char c)
  {
    return (isspace(c) || c == '\n' || c == '\t' || c == ',');
  }
};

} // namespace lbann

#endif // LBANN_DATA_READER_SMILES_HPP
