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

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

// The code being tested
#include "lbann/data_readers/data_reader_smiles.hpp"

#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/random_number_generators.hpp"

#include <lbann.pb.h>

#include <google/protobuf/text_format.h>

#include <ctime>
#include <string>

// input data
#include "test_data/A_smiles_reader.smi"
#include "test_data/B_smiles_reader.smi"
#include "test_data/C_smiles_reader.smi"
#include "test_data/D_smiles_reader.smi"
#include "test_data/smiles_reader.prototext"
#include "test_data/smiles_reader_sample_list.txt"
#include "test_data/vocab.txt"

namespace pb = ::google::protobuf;
namespace utils = ::unit_test::utilities;
using lbann::file::join_path;

// Make a temporary directory with the given name.
std::string get_tmpdir() noexcept;

// compute offsets and lengths and writes to binary file;
// returns the number of sequences in the file
int write_offsets(std::string const& smi,
                  std::string const& output_fn,
                  std::string const& tmp_dir);

// write smiles strings to file: /tmp/<filename>
void write_smiles_data_to_file(std::string const& smi,
                               std::string const& output_fn,
                               std::string const& tmp_dir);

void test_fetch(lbann::generic_data_reader* reader);

TEST_CASE("SMILES functional black-box",
          "[.filesystem][data_reader][mpi][smiles]")
{
  auto& comm = utils::current_world_comm();
  lbann::init_data_seq_random(42);

  // The data reader behavior depends on arguments passed on the
  // command line (super...). Therefore, we should restore its state
  // to the expected state/
  auto& arg_parser = utils::reset_global_argument_parser();

  // Get a temporary location in the file system (:/).
  std::string const tmp_dir = get_tmpdir();
  REQUIRE_NOTHROW(lbann::file::make_directory(tmp_dir));

  // test that we can write files to the tmp_dir
  {
    std::ofstream out(join_path(tmp_dir, "test"));
    REQUIRE(out.good());
  }

  // write binary offset files
  int const n_seqs_A =
    write_offsets(A_smi_const, "A_smiles_reader.offsets", tmp_dir);
  int const n_seqs_B =
    write_offsets(B_smi_const, "B_smiles_reader.offsets", tmp_dir);
  int const n_seqs_C =
    write_offsets(C_smi_const, "C_smiles_reader.offsets", tmp_dir);
  int const n_seqs_D =
    write_offsets(D_smi_const, "D_smiles_reader.offsets", tmp_dir);

  // copy smiles data and metadata file
  write_smiles_data_to_file(A_smi_const, "A_smiles_reader.smi", tmp_dir);
  write_smiles_data_to_file(B_smi_const, "B_smiles_reader.smi", tmp_dir);
  write_smiles_data_to_file(C_smi_const, "C_smiles_reader.smi", tmp_dir);
  write_smiles_data_to_file(D_smi_const, "D_smiles_reader.smi", tmp_dir);

  // === START: fix place-holders in the prototext, then write to file
  // adjust prototext "label_filename" to point to correct metadata file
  size_t j1 = smiles_reader_prototext.find("METADATA_FN");
  REQUIRE(j1 != std::string::npos);
  std::string replacement = join_path(tmp_dir, "metadata");
  smiles_reader_prototext.replace(j1, 11, replacement);

  // adjust prototext "sample_list" to point to correct sample list file
  j1 = smiles_reader_prototext.find("SAMPLE_LIST_FN");
  REQUIRE(j1 != std::string::npos);
  replacement = join_path(tmp_dir, "sample_list");
  smiles_reader_prototext.replace(j1, 14, replacement);

  // write the prototext file
  std::string const prototext_fn = join_path(tmp_dir, "prototext");
  {
    std::ofstream out(prototext_fn);
    REQUIRE(out.good());
    out << smiles_reader_prototext << '\n';
  }
  // === END: fix place-holders in the prototex, then write to file

  // construct metadata file contents
  {
    std::ofstream meta(join_path(tmp_dir, "metadata"));
    REQUIRE(meta.good());
    meta << n_seqs_A << " " << join_path(tmp_dir, "A_smiles_reader.smi") << ' '
         << join_path(tmp_dir, "A_smiles_reader.offsets") << '\n'
         << n_seqs_D << ' ' << join_path(tmp_dir, "D_smiles_reader.smi") << ' '
         << join_path(tmp_dir, "D_smiles_reader.offsets") << '\n'
         << n_seqs_B << ' ' << join_path(tmp_dir, "B_smiles_reader.smi") << ' '
         << join_path(tmp_dir, "B_smiles_reader.offsets") << '\n'
         << n_seqs_C << ' ' << join_path(tmp_dir, "C_smiles_reader.smi ")
         << join_path(tmp_dir, "C_smiles_reader.offsets") << '\n';
  }

  // write the vocabulary file. The filename is needed later to pass
  // to the argument parser.
  auto const& vocab_fn = join_path(tmp_dir, "vocab.txt");
  {
    std::ofstream out(vocab_fn);
    REQUIRE(out.good());
    out << vocab_txt_const; // from: #include "test_data/vocab.txt"
  }
  std::clog << "-- wrote: " << vocab_fn << std::endl;

  // adjust "BASE_DIR" place-holder, then write the sample list file
  {
    auto const sample_list_filename(join_path(tmp_dir, "sample_list"));
    std::ofstream out(sample_list_filename);
    REQUIRE(out.good());

    replacement = tmp_dir;
    j1 = sample_list.find("BASE_DIR");
    REQUIRE(j1 != std::string::npos);
    sample_list.replace(j1, 8, replacement);
    out << sample_list << '\n';
    std::clog << "-- wrote: " << sample_list_filename << std::endl;
  }

  //=========================================================================
  // instantiate and setup the data reader
  //=========================================================================
  lbann_data::LbannPB my_proto;
  REQUIRE(pb::TextFormat::ParseFromString(smiles_reader_prototext, &my_proto));

  // set up the options that the reader expects
  char const* const argv[] = {"smiles_functional_black_box.exe",
                              "--use_data_store",
                              "--preload_data_store",
                              "--sequence_length=100",
                              "--vocab",
                              vocab_fn.c_str()};
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
    REQUIRE(unit_test::utilities::IsValidPtr(t.second));
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

  // Cleanup the data readers
  for (auto t : data_readers) {
    delete t.second;
  }
}

int write_offsets(std::string const& smi,
                  std::string const& output_fn,
                  std::string const& tmp_dir)
{
  int n_seqs = 0;

  // open output file
  std::ofstream out(join_path(tmp_dir, output_fn), std::ios::binary);
  LBANN_ASSERT(out.good());

  // compute and write contents of offset file for the input SMILES (smi)
  // string; at least one input data file (D_smiles_reader.smi) should have
  // at least one of each of the legal delimiters: space, comma, tab, newline
  std::stringstream ss(smi);
  char buf[2048];
  do {
    long long start = (long long)ss.tellg();
    ss.getline(buf, 2048);
    std::string smiles_str(buf);

    // find the delimiter, which determines the length of the sequence;
    // check for one of the delimiters: tab, comma, space, newline;
    // implicit assumption: these will never be characters in a valid
    // SMILES string
    size_t const len_1 = smiles_str.find(" ");
    size_t const len_2 = smiles_str.find(",");
    size_t const len_3 = smiles_str.find("\t");
    size_t const len_4 = smiles_str.size();
    LBANN_ASSERT(len_1 > 0UL);
    LBANN_ASSERT(len_2 > 0UL);
    LBANN_ASSERT(len_3 > 0UL);
    LBANN_ASSERT(len_4 > 0UL);
    size_t length = SIZE_MAX;
    if (len_1 < length)
      length = len_1;
    if (len_2 < length)
      length = len_2;
    if (len_3 < length)
      length = len_3;
    if (len_4 < length)
      length = len_4;
    LBANN_ASSERT(length != SIZE_MAX);

    short const len = length;
    out.write((char*)&start, sizeof(long long));
    out.write((char*)&len, sizeof(short));
    ++n_seqs;
  } while (ss.good() && !ss.eof());

  return n_seqs;
}

void write_smiles_data_to_file(std::string const& smi,
                               std::string const& output_fn,
                               std::string const& tmp_dir)
{
  auto const outfile = join_path(tmp_dir, output_fn);
  std::ofstream out(outfile, std::ios::binary);
  LBANN_ASSERT(out.good());
  out << smi;
  std::clog << "-- wrote: " << outfile << std::endl;
}

void test_fetch(lbann::generic_data_reader* reader)
{
  LBANN_ASSERT(unit_test::utilities::IsValidPtr(reader));
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
    std::clog << "-- getting origin for: " << index << std::endl;
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

std::string get_tmpdir() noexcept
{
  std::string tmpdir;
  if (auto const* tmp = std::getenv("TMPDIR"))
    tmpdir = tmp;
  else
    tmpdir = "/tmp";
  return join_path(
    tmpdir,
    lbann::build_string("smiles_fetch_datum_test_", std::time(nullptr)));
}
