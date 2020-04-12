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

#ifndef LBANN_DATA_READER_SMILES_HPP
#define LBANN_DATA_READER_SMILES_HPP

#include "conduit/conduit.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/data_readers/data_reader.hpp"

namespace lbann {
  /**
   * Data reader for SMILES data that has been converted to an array
   * of short ints and stored in binary format.
   * Binary format is: (n_int, int (repeating n_int times) ) repeating
   * last entry in the file is the only entry stored as an integer; it
   * contains the number of samples. Second to last entry is the maximum
   * number of ints in any sample; this is stored as a short int
   */
class smiles_data_reader : public generic_data_reader {

public:

  smiles_data_reader(const bool shuffle);
  smiles_data_reader(const smiles_data_reader&);
  smiles_data_reader& operator=(const smiles_data_reader&);
  ~smiles_data_reader() override;

  smiles_data_reader* copy() const override { return new smiles_data_reader(*this); }

  std::string get_type() const override {
    return "smiles_data_reader";
  }

  void load() override;

  int get_linearized_data_size() const override { return m_linearized_data_size; }
  int get_linearized_label_size() const override {  return m_linearized_label_size; }
  int get_linearized_response_size() const override { return m_linearized_response_size; }
  const std::vector<int> get_data_dims() const override {  return {get_linearized_data_size()}; }
  int get_num_labels() const override { return m_num_labels; }

  void set_sequence_length(int n) { m_linearized_data_size = n; }
  int get_sequence_length() { return get_linearized_data_size(); }

  void set_num_samples(int n) { m_num_samples = n; }
  int get_num_samples() { return m_num_samples; }

private:

  int m_num_samples = -1;

  std::ifstream m_data_stream;

  int m_linearized_data_size = 0;
  int m_linearized_label_size = 0;
  int m_linearized_response_size = 0;
  int m_num_labels = 0;

  //TODO: make this a user setting -- ??
  const short m_pad = 420;

  const short m_unk = 421;

  std::unordered_map<char, short> m_vocab;

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
  void load_vocab();
  int get_num_lines(std::string fn); 
  void construct_conduit_node(int data_id, const std::string &line, conduit::Node &node); 
  void encode_smiles(const std::string &sm, std::vector<short> &data);

};

}  // namespace lbann

#endif //LBANN_DATA_READER_SMILES_HPP
