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
////////////////////////////////////////////////////////////////////////////////

#include "Catch2BasicSupport.hpp"

#include "TestHelpers.hpp"

#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>

// The code being tested
#include "lbann/data_ingestion/readers/data_reader_smiles.hpp"

namespace pb = ::google::protobuf;

TEST_CASE("SMILES string encoder", "[data_reader][smiles]")
{
  // For what it's worth, smiles_data_reader::decode_smiles() isn't
  // used during normal operations (training, inferencing).

  const int data_size_normal(100);
  const int data_size_small(5);

  std::stringstream vocab(
    "# 0 % 1 ( 2 ) 3 + 4 - 5 . 6 / 7 0 8 1 9 2 10 3 11 4 12 5 13 6 14 7 15 8 "
    "16 9 17 = 18 @ 19 B 20 C 21 F 22 H 23 I 24 N 25 O 26 P 27 S 28 [ 29 \\ 30 "
    "] 31 c 32 e 33 i 34 l 35 n 36 o 37 p 38 r 39 s 40 <bos> 41 <eos> 42 <pad> "
    "43 <unk> 44");

  auto smiles = std::make_unique<lbann::smiles_data_reader>(true);
  smiles->load_vocab(vocab);

  const std::string smi_1("C#CCCNC1=NN(C)C=C1");  // good
  const std::string smi_2("C#CCCNC1 NN(C)C=C1");  // space in middle
  const std::string smi_3(" CC(C(=O)NN)C1=CC=");  // space at beginning
  const std::string smi_4("C[C@H](CCO)NC1CCSC,"); // comma at end

  SECTION("encode+decode")
  {
    smiles->set_linearized_data_size(data_size_normal);

    std::vector<unsigned short> encoded;
    std::string decoded;

    bool r1 = smiles->encode_smiles(smi_1, encoded);
    CHECK(r1);
    smiles->decode_smiles(encoded, decoded);
    CHECK(smi_1 == decoded);

    bool r2 = smiles->encode_smiles(smi_2, encoded);
    CHECK(!r2);
    smiles->decode_smiles(encoded, decoded);
    CHECK(smi_2 != decoded);

    bool r3 = smiles->encode_smiles(smi_3, encoded);
    CHECK(!r3);
    smiles->decode_smiles(encoded, decoded);
    CHECK(smi_3 != decoded);

    bool r4 = smiles->encode_smiles(smi_4, encoded);
    CHECK(!r4);
    smiles->decode_smiles(encoded, decoded);
    CHECK(smi_4 != decoded);
  }
  SECTION("encode+truncate")
  {
    // test for input sequences longer than max permitted sequence length
    smiles->set_linearized_data_size(data_size_small);
    std::vector<unsigned short> encoded;
    std::string decoded;
    bool r1 = smiles->encode_smiles(smi_1, encoded);
    CHECK(r1);
    CHECK(encoded.size() == data_size_small);
    smiles->decode_smiles(encoded, decoded);
    CHECK(smi_1.find(decoded) != std::string::npos);
  }
  SECTION("decode+bad character")
  {
    // test for input sequences longer than max permitted sequence length
    smiles->set_linearized_data_size(data_size_normal);

    std::vector<unsigned short> encoded;
    std::string decoded;
    bool r1 = smiles->encode_smiles(smi_1, encoded);
    CHECK(r1);
    encoded[2] = 99;
    REQUIRE_THROWS(smiles->decode_smiles(encoded, decoded));
    std::cout << decoded << std::endl;
  }
}

TEST_CASE("SMILES istream reader", "[data_reader][smiles]")
{
  auto smiles = std::make_unique<lbann::smiles_data_reader>(true);

  std::stringstream data;
  data << "COC1=C(NC(=O)C( s_2\n"
          "NC1=NC=NC(NCC(F)   \n"
          " CCN([C@@H] s_22_nu\n";
  int index = 0;
  smiles->set_offset(index++, 0, 15);
  smiles->set_offset(index++, 20, 16);
  smiles->set_offset(index++, 30, 11);

  size_t buf_offset = 0;

  SECTION("good strings")
  {
    std::string sample;
    sample = smiles->get_raw_sample(&data, 0, buf_offset);
    CHECK(sample == "COC1=C(NC(=O)C(");
    sample = smiles->get_raw_sample(&data, 1, buf_offset);
    CHECK(sample == "NC1=NC=NC(NCC(F)");
  }
  SECTION("bad strings")
  {
    // space at char preceeding string
    REQUIRE_THROWS(smiles->get_raw_sample(&data, 2, buf_offset));

    // set length too short
    smiles->set_offset(0, 0, 10);
    REQUIRE_THROWS(smiles->get_raw_sample(&data, 0, buf_offset));
  }
}

TEST_CASE("SMILES ingestion - REAL/S/H30/SH30M600.smi", "[data_reader][smiles]")
{
  auto smiles = std::make_unique<lbann::smiles_data_reader>(true);

  /// /p/vast1/atom/arthor_dbs/REAL/S/H30/SH30M600.smi
  std::string const SH30M600_smi =
    R"smile(CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)N2CCNC(=O)C2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____17553516____16656074
CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)N2CC(=O)NCC2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____14789470____16656074)smile";

  size_t const line_len = 111;
  size_t const valid_chars = 82;

  std::string const smiles_str = SH30M600_smi;
  std::istringstream iss(smiles_str);
  smiles->set_offset(0, 0, line_len);
  smiles->set_offset(1, line_len + 1, line_len);

  SECTION("Sample 0")
  {
    std::string str = smiles->get_raw_sample(&iss, 0, 0);
    CHECK(str == smiles_str.substr(0, valid_chars));
  }
  SECTION("Sample 1")
  {
    std::string str = smiles->get_raw_sample(&iss, 1, 0);
    CHECK(str == smiles_str.substr(line_len + 1, valid_chars));
  }
}

TEST_CASE("SMILES ingestion - malformed REAL/S/H30/SH30M600.smi",
          "[data_reader][smiles]")
{
  lbann::smiles_data_reader* smiles = new lbann::smiles_data_reader(true);

  /// /p/vast1/atom/arthor_dbs/REAL/S/H30/SH30M600.smi - malformed with
  /// a , and a ' ' inserted into each string
  std::string const SH30M600_smi_malformed =
    R"smile(CC(=O)N[C@@H]1[C@@H](O)C[C@@](O),(C(=O)N2CCNC(=O)C2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____17553516____16656074
CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)N2CC (=O)NCC2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____14789470____16656074)smile";

  size_t const line_len = 112;
  size_t const sample_one_valid_chars = 32;
  size_t const sample_two_valid_chars = 42;

  std::string const smiles_str = SH30M600_smi_malformed;
  std::istringstream iss(smiles_str);
  smiles->set_offset(0, 0, line_len);
  smiles->set_offset(1, line_len + 1, line_len);
  SECTION("Sample 0")
  {
    std::string str = smiles->get_raw_sample(&iss, 0, 0);
    CHECK(str == smiles_str.substr(0, sample_one_valid_chars));
  }
  SECTION("Sample 1")
  {
    std::string str = smiles->get_raw_sample(&iss, 1, 0);
    CHECK(str == smiles_str.substr(line_len + 1, sample_two_valid_chars));
  }
}
