////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include <catch2/catch.hpp>

#include "TestHelpers.hpp"
#include "MPITestHelpers.hpp"

#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/random_number_generators.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/lbann_library.hpp"

#include "lbann/proto/proto_common.hpp"
#include <lbann.pb.h>
#include <google/protobuf/text_format.h>

// The code being tested
#include "lbann/data_readers/data_reader_smiles.hpp"
#include "lbann/data_store/data_store_conduit.hpp"

#include <sys/types.h> //for getpid
#include <unistd.h>    //for getpid
#include <sys/stat.h>    //for mkdir
#include <sys/types.h>   //for mkdir
#include <errno.h>
#include <string.h>
#include <cstdlib>

// input data
#include "test_data/A_smiles_reader.smi"
#include "test_data/B_smiles_reader.smi"
#include "test_data/C_smiles_reader.smi"
#include "test_data/D_smiles_reader.smi"
#include "test_data/smiles_reader.prototext"
#include "test_data/vocab.txt"
#include "test_data/smiles_reader_sample_list.txt"

namespace pb = ::google::protobuf;

// compute offsets and lengths and writes to binary file;
// returns the number of sequences in the file
int write_offsets(const std::string& smi, const std::string output_fn, const char *tmp_dir);

// write smiles strings to file: /tmp/<filename>
void write_smiles_data_to_file(const std::string smi, const std::string output_fn, const char* tmp_dir);

void test_fetch(lbann::generic_data_reader* reader);

bool directory_exists(std::string s);

TEST_CASE("SMILES functional black-box", "[.filesystem][data reader][mpi][smiles]")
{
  //currently, tests are sequential; they can (should?) be expanded
  //to multiple ranks

  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);
  auto& arg_parser = lbann::global_argument_parser();
  arg_parser.clear(); // Clear the argument parser.
  lbann::construct_all_options();

  //make non-const copies
  std::string smiles_reader_prototext(smiles_reader_prototext_const);
  std::string sample_list(sample_list_const);

  //=========================================================================
  // create directory: /tmp/smiles_reader_test_<pid>,
  // then write the input files that the reader expects,
  // during normal operation of a network with the lbann executable
  //=========================================================================
  pid_t pid = getpid();
  char b[2048];
  sprintf(b, "/tmp/smiles_reader_test_%d", pid);
  const std::string tmp_dir(b);
  const char *tdir = tmp_dir.data();

  //create working directory; this will contain the inputs files that
  //the smiles_data_reader expects during load()
  lbann::file::make_directory(tmp_dir);

  //test that we can write files to the tmp_dir
  sprintf(b, "%s/test", tdir);
  std::ofstream out(b);
  REQUIRE(out.good());
  out.close();

  //write binary offset files
  int n_seqs_B = write_offsets(B_smi_const, "B_smiles_reader.offsets", tdir);
  int n_seqs_A = write_offsets(A_smi_const, "A_smiles_reader.offsets", tdir);
  int n_seqs_C = write_offsets(C_smi_const, "C_smiles_reader.offsets", tdir);
  int n_seqs_D = write_offsets(D_smi_const, "D_smiles_reader.offsets", tdir);

  //copy smiles data and metadata file
  write_smiles_data_to_file(C_smi_const, "C_smiles_reader.smi", tdir);
  write_smiles_data_to_file(B_smi_const, "B_smiles_reader.smi", tdir);
  write_smiles_data_to_file(A_smi_const, "A_smiles_reader.smi", tdir);
  write_smiles_data_to_file(D_smi_const, "D_smiles_reader.smi", tdir);

  // === START: fix place-holders in the prototex, then write to file
  // adjust prototext "label_filename" to point to correct metadata file
  size_t j1 = smiles_reader_prototext.find("METADATA_FN");
  REQUIRE(j1 != std::string::npos);
  sprintf(b, "%s/metadata", tdir);
  std::string replacement(b);
  smiles_reader_prototext.replace(j1, 11, replacement);

  // adjust prototext "sample_list" to point to correct sample list file
  j1 = smiles_reader_prototext.find("SAMPLE_LIST_FN");
  REQUIRE(j1 != std::string::npos);
  sprintf(b, "%s/sample_list", tdir);
  replacement = b;
  smiles_reader_prototext.replace(j1, 14, replacement);

  // write the prototext file
  sprintf(b, "%s/prototext", tdir);
  std::string prototext_fn(b);
  out.open(prototext_fn);
  REQUIRE(out.good());
  out << smiles_reader_prototext << std::endl;
  out.close();
  // === END: fix place-holders in the prototex, then write to file

  // construct metadata file contents
  std::stringstream meta;
  sprintf(b, "%s/", tdir);
  meta << n_seqs_A << " " << b
       << "A_smiles_reader.smi " << b << "A_smiles_reader.offsets" << std::endl
       << n_seqs_D << " " << b
       << "D_smiles_reader.smi " << b << "D_smiles_reader.offsets" << std::endl
       << n_seqs_B << " " << b
       << "B_smiles_reader.smi " << b << "B_smiles_reader.offsets" << std::endl
       << n_seqs_C << " " << b
       << "C_smiles_reader.smi " << b << "C_smiles_reader.offsets" << std::endl;

  // write the metadata file
  sprintf(b, "%s/metadata", tdir);
  std::string metadata_fn(b);
  out.open(metadata_fn);
  REQUIRE(out.good());
  out << meta.str();
  out.close();

  //write the vocabulary file
  sprintf(b, "%s/vocab.txt", tdir);
  const std::string vocab_fn(b);
  out.open(b);
  REQUIRE(out.good());
  out << vocab_txt_const; //from: #include "test_data/vocab.txt"
  out.close();
  std::cout << "wrote: " << b << std::endl;

  //adjust "BASE_DIR" place-holder, then write the sample list file
  sprintf(b, "%s/sample_list", tdir);
  std::string sample_list_filename(b);
  out.open(sample_list_filename);
  REQUIRE(out.good());
  sprintf(b, "%s/", tdir);
  replacement = b;
  j1 = sample_list.find("BASE_DIR");
  REQUIRE(j1 != std::string::npos);
  sample_list.replace(j1, 8, replacement);
  out << sample_list << std::endl;
  out.close();
  std::cout << "wrote: " << sample_list_filename << std::endl;

  //=========================================================================
  // instantiate and setup the data reader
  //=========================================================================
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(smiles_reader_prototext, &my_proto)) {
    throw "Parsing protobuf failed.";
  }

  // set up the options that the reader expects
  char const* argv[] = {"smiles_functional_black_box.exe",
    "--use_data_store",
    "--preload_data_store",
    "--sequence_length=100",
    "--vocab", vocab_fn.c_str()};
  int const argc = sizeof(argv) / sizeof(argv[0]);
  REQUIRE_NOTHROW(arg_parser.parse(argc, argv));

  // instantiate and load the data readers
  std::map<lbann::execution_mode, lbann::generic_data_reader*> data_readers;
  lbann::init_data_readers(&comm, my_proto, data_readers);

  // get pointers the the various readers
  lbann::generic_data_reader* train_ptr = nullptr;
  lbann::generic_data_reader* validate_ptr = nullptr;
  lbann::generic_data_reader* test_ptr = nullptr;
  for (auto t : data_readers) {
    if (t.second->get_role() == "train") {
      train_ptr = t.second;
    }
    if (t.second->get_role() == "validate") {
      validate_ptr = t.second;
    }
    if (t.second->get_role() == "test") {
      test_ptr = t.second;
    }
  }


  SECTION("compare disk against data_store SMILES strings")
  {
    if (train_ptr) {
      test_fetch(train_ptr);
    }
    if (validate_ptr) {
      test_fetch(validate_ptr);
    }
    if (test_ptr) {
      test_fetch(test_ptr);
    }
  }

  arg_parser.clear(); // Clear the argument parser.
  // Cleanup the data readers
  for (auto t : data_readers) {
    delete t.second;
  }
}

int write_offsets(const std::string& smi, const std::string output_fn, const char* tmp_dir) {
  int n_seqs = 0;

  //open output file
  char b[2048];
  sprintf(b, "%s/%s", tmp_dir, output_fn.c_str());
  std::ofstream out(b, std::ios::binary);
  REQUIRE(out.good());

  // compute and write contents of offset file for the input SMILES (smi)
  // string; at least one input data file (D_smiles_reader.smi) should have
  // at least one of each of the legal delimiters: space, comma, tab, newline
  std::stringstream ss(smi);
  char buf[2048];
  do {
    long long start = (long long)ss.tellg();
    ss.getline(buf, 2048);
    std::string smiles_str(buf);

    //find the delimiter, which determines the length of the sequence;
    //check for one of the delimiters: tab, comma, space, newline;
    //implicit assumption: these will never be characters in a valid
    //SMILES string
    size_t len_1 = smiles_str.find(" ");
    size_t len_2 = smiles_str.find(",");
    size_t len_3 = smiles_str.find("\t");
    size_t len_4 = smiles_str.size();
    REQUIRE(len_1);
    REQUIRE(len_2);
    REQUIRE(len_3);
    REQUIRE(len_4);
    size_t length = SIZE_MAX;
    if (len_1 < length) length = len_1;
    if (len_2 < length) length = len_2;
    if (len_3 < length) length = len_3;
    if (len_4 < length) length = len_4;
    REQUIRE(length != SIZE_MAX);

    short len = length;
    out.write((char*)&start, sizeof(long long));
    out.write((char*)&len, sizeof(short));
    ++n_seqs;
  } while (ss.good() && !ss.eof());

  out.close();
  return n_seqs;
}

void write_smiles_data_to_file(const std::string smi, const std::string output_fn, const char* tmp_dir) {
  char b[2048];
  sprintf(b, "%s/%s", tmp_dir, output_fn.c_str());
  std::ofstream out(b, std::ios::binary);
  REQUIRE(out.good());
  out << smi;
  out.close();
  std::cout << "wrote: " << b << std::endl;
}

void test_fetch(lbann::generic_data_reader* reader) {
  reader->preload_data_store();
#if 0
  const std::vector<int>& indices = reader->get_shuffled_indices();

  lbann::smiles_data_reader *r = dynamic_cast<lbann::smiles_data_reader*>(reader);
  REQUIRE(r != nullptr);
  std::string fn;
  size_t offset;
  short length;


  // this mimics a call to smiles_data_reader::fetch_datum;
  // however, it doesn't test inserting the data into a matrix
  // (which, some would say, should not be the task of a data reader, anyway)
  for (auto index : indices) {
    const conduit::Node& node = reader->get_data_store_ptr()->get_conduit_node(index);
    const conduit::unsigned_short_array tmp_data = node["/" + LBANN_DATA_ID_STR(index) + "/data"].as_unsigned_short_array();
    std::vector<unsigned short> data;
    size_t n = tmp_data.number_of_elements();
    for (size_t j=0; j<n; j++) {
      data.push_back(tmp_data[j]);
    }
    //at this point, 'data' contains the encoded smiles string;
    //note that this may contain fewer entries than the original
    //smiles string; this will happen if sequence_length is less than
    // a smiles string length.

    //decode data back to a string
    std::string decoded;
    r->decode_smiles(data, decoded);

    // read the smiles string from file
    std::cout << "getting origin for: " << index << std::endl;
    r->get_sample_origin(index, fn, offset, length);
    std::ifstream in(fn.c_str());
    REQUIRE(in);
    in.seekg(offset);
    REQUIRE(in.good());
    std::string from_file;
    from_file.resize(length);
    in.read((char*)from_file.data(), length);
    in.close();
    REQUIRE(decoded == from_file);

/*
    std::vector<unsigned short> f_test;
    r->encode_smiles(from_file, f_test);
    std::string f_test_str;
    r->decode_smiles(f_test, f_test_str);
    REQUIRE(decoded == from_file);
*/
  }
  #endif
}

bool directory_exists(std::string s) {
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}
