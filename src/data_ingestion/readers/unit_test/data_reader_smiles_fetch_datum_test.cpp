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

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

// The code being tested
#include "lbann/data_ingestion/readers/data_reader_smiles.hpp"

#include "lbann/comm.hpp"
#include "lbann/proto/proto_common.hpp" // init_data_readers
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"

#include "lbann/proto/lbann.pb.h"

#include <google/protobuf/text_format.h>

#include <ctime> // Use time since epoch as unique tmp directory.
#include <regex>
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

// Get the name of a temporary directory.
static std::string get_tmpdir() noexcept;

// Do all the setup of the filetree rooted at tmp_dir:
//
//   tmp_dir/A_smiles_reader.offsets
//   tmp_dir/A_smiles_reader.smi
//   tmp_dir/B_smiles_reader.offsets
//   tmp_dir/B_smiles_reader.smi
//   tmp_dir/C_smiles_reader.offsets
//   tmp_dir/C_smiles_reader.smi
//   tmp_dir/D_smiles_reader.offsets
//   tmp_dir/D_smiles_reader.smi
//   tmp_dir/metadata
//   tmp_dir/prototext
//   tmp_dir/sample_list
//   tmp_dir/vocab.txt
static void setup_tmp_filetree(std::string const& tmp_dir);

static void test_fetch(lbann::generic_data_reader& reader);

TEST_CASE("SMILES functional black-box",
          "[.filesystem][data_reader][mpi][smiles]")
{
  auto& comm = utils::current_world_comm();
  lbann::init_data_seq_random(42);

  // Setup some filesystem stuff
  auto const tmp_dir = get_tmpdir();
  REQUIRE_NOTHROW(setup_tmp_filetree(tmp_dir));

  // The data reader behavior depends on arguments passed on the
  // command line (super...). Therefore, we should restore its state
  // to the expected state.
  auto& arg_parser = utils::reset_global_argument_parser();
  {
    auto const vocab_fn = join_path(tmp_dir, "vocab.txt");
    char const* const argv[] = {"smiles_functional_black_box.exe",
                                "--use_data_store",
                                "--preload_data_store",
                                "--sequence_length=100",
                                "--vocab",
                                vocab_fn.c_str()};
    int const argc = sizeof(argv) / sizeof(argv[0]);
    REQUIRE_NOTHROW(arg_parser.parse(argc, argv));
  }

  // Parse the data reader prototext
  lbann_data::LbannPB my_proto;
  REQUIRE(pb::TextFormat::ParseFromString(smiles_reader_prototext, &my_proto));

  // Instantiate and load the data readers
  std::map<lbann::execution_mode, std::shared_ptr<lbann::generic_data_reader>>
    data_readers;
  {
    std::map<lbann::execution_mode, lbann::generic_data_reader*>
      data_readers_tmp;
    lbann::init_data_readers(&comm, my_proto, data_readers_tmp);
    for (auto& [mode, dr] : data_readers_tmp)
      data_readers[mode].reset(dr);
  }

  // get pointers the the various readers
  lbann::generic_data_reader* train_ptr = nullptr;
  lbann::generic_data_reader* validate_ptr = nullptr;
  lbann::generic_data_reader* test_ptr = nullptr;
  for (auto& [mode, dr] : data_readers) {
    (void)mode;
    auto* raw_dr_ptr = dr.get();
    REQUIRE(unit_test::utilities::IsValidPtr(raw_dr_ptr));
    if (dr->get_role() == "train") {
      train_ptr = raw_dr_ptr;
    }
    if (dr->get_role() == "validate") {
      validate_ptr = raw_dr_ptr;
    }
    if (dr->get_role() == "test") {
      test_ptr = raw_dr_ptr;
    }
  }

  SECTION("compare disk against data_store SMILES strings")
  {
    if (train_ptr) {
      REQUIRE_NOTHROW(test_fetch(*train_ptr));
    }
    if (validate_ptr) {
      REQUIRE_NOTHROW(test_fetch(*validate_ptr));
    }
    if (test_ptr) {
      REQUIRE_NOTHROW(test_fetch(*test_ptr));
    }
  }
}

static int write_offsets(std::string const& smi, std::ostream& out)
{
  // Assumption: These are not valid characters in a SMILES string.
  std::string const delimiters = " ,\t";

  // compute and write contents of offset file for the input SMILES (smi)
  // string; at least one input data file (D_smiles_reader.smi) should have
  // at least one of each of the legal delimiters: space, comma, tab, newline
  std::istringstream iss;
  iss.str(smi);
  int n_seqs = 0;
  for (std::string smiles_str; iss;) {
    long long const start = (long long)iss.tellg();
    std::getline(iss, smiles_str);

    // Check for a delimiter (tab, comma, space, newline). Everything
    // after the delimiter is ignored.
    short const length = static_cast<short>(
      std::min(smiles_str.find_first_of(delimiters), smiles_str.size()));
    out.write((char*)&start, sizeof(long long));
    out.write((char*)&length, sizeof(short));
    ++n_seqs;
  }
  return n_seqs;
}

static int write_offsets_to_file(std::string const& smi,
                                 std::string const& output_fn,
                                 std::string const& tmp_dir)
{
  std::ofstream out(join_path(tmp_dir, output_fn), std::ios::binary);
  LBANN_ASSERT(out.good());
  return write_offsets(smi, out);
}

static void write_smiles_data_to_file(std::string const& smi,
                                      std::string const& output_fn,
                                      std::string const& tmp_dir)
{
  std::ofstream out(join_path(tmp_dir, output_fn), std::ios::binary);
  LBANN_ASSERT(out.good());
  out << smi;
}

static void write_prototext_to_file(std::string const& ptext,
                                    std::string const& output_fn,
                                    std::string const& tmp_dir)
{
  std::ofstream out(join_path(tmp_dir, "prototext"));
  LBANN_ASSERT(out.good());
  out << ptext << '\n';
}

static void write_vocab_to_file(std::string const& vocab,
                                std::string const& output_fn,
                                std::string const& tmp_dir)
{
  std::ofstream out(join_path(tmp_dir, output_fn));
  REQUIRE(out.good());
  out << vocab; // from: #include "test_data/vocab.txt"
}

static void write_metadata_file(int const n_seqs_A,
                                int const n_seqs_B,
                                int const n_seqs_C,
                                int const n_seqs_D,
                                std::string const& output_fn,
                                std::string const& tmp_dir)
{
  std::ofstream meta(join_path(tmp_dir, output_fn));
  LBANN_ASSERT(meta.good());
  meta << n_seqs_A << " " << join_path(tmp_dir, "A_smiles_reader.smi") << ' '
       << join_path(tmp_dir, "A_smiles_reader.offsets") << '\n'
       << n_seqs_D << ' ' << join_path(tmp_dir, "D_smiles_reader.smi") << ' '
       << join_path(tmp_dir, "D_smiles_reader.offsets") << '\n'
       << n_seqs_B << ' ' << join_path(tmp_dir, "B_smiles_reader.smi") << ' '
       << join_path(tmp_dir, "B_smiles_reader.offsets") << '\n'
       << n_seqs_C << ' ' << join_path(tmp_dir, "C_smiles_reader.smi ")
       << join_path(tmp_dir, "C_smiles_reader.offsets") << '\n';
}

static void write_sample_list_to_file(std::string const& slist,
                                      std::string const& output_fn,
                                      std::string const& tmp_dir)
{
  std::ofstream out(join_path(tmp_dir, "sample_list"));
  LBANN_ASSERT(out.good());
  out << slist << '\n';
}

static void test_fetch(lbann::generic_data_reader& reader)
{
  reader.preload_data_store();
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

static std::string get_tmpdir() noexcept
{
  std::string tmpdir;
  if (auto const* tmp = std::getenv("TMPDIR"))
    tmpdir = tmp;
  else
    tmpdir = "/tmp";
  return join_path(tmpdir,
                   lbann::build_string("smiles_fetch_datum_test_",
                                       std::time(nullptr),
                                       "_",
                                       lbann::get_rank_in_world()));
}

static void setup_tmp_filetree(std::string const& tmp_dir)
{
  // Get a temporary location in the file system (:/).
  lbann::file::make_directory(tmp_dir);
  std::clog << "-- Created tmpdir: " << tmp_dir << std::endl;

  // test that we can write files to the tmp_dir
  {
    std::ofstream out(join_path(tmp_dir, "test"));
    LBANN_ASSERT(out.good());
  }

  // write binary offset files
  int const n_seqs_A =
    write_offsets_to_file(A_smi_const, "A_smiles_reader.offsets", tmp_dir);
  int const n_seqs_B =
    write_offsets_to_file(B_smi_const, "B_smiles_reader.offsets", tmp_dir);
  int const n_seqs_C =
    write_offsets_to_file(C_smi_const, "C_smiles_reader.offsets", tmp_dir);
  int const n_seqs_D =
    write_offsets_to_file(D_smi_const, "D_smiles_reader.offsets", tmp_dir);

  // copy smiles data and metadata file
  write_smiles_data_to_file(A_smi_const, "A_smiles_reader.smi", tmp_dir);
  write_smiles_data_to_file(B_smi_const, "B_smiles_reader.smi", tmp_dir);
  write_smiles_data_to_file(C_smi_const, "C_smiles_reader.smi", tmp_dir);
  write_smiles_data_to_file(D_smi_const, "D_smiles_reader.smi", tmp_dir);

  // Fix placeholders in the input data.
  std::regex const tmpdir_re("TMPDIR");
  smiles_reader_prototext =
    std::regex_replace(smiles_reader_prototext, tmpdir_re, tmp_dir);
  sample_list = std::regex_replace(sample_list, tmpdir_re, tmp_dir);

  // write the required files
  write_prototext_to_file(smiles_reader_prototext, "prototext", tmp_dir);
  std::clog << "-- wrote protobuf file: " << join_path(tmp_dir, "prototext")
            << std::endl;

  write_metadata_file(n_seqs_A,
                      n_seqs_B,
                      n_seqs_C,
                      n_seqs_D,
                      "metadata",
                      tmp_dir);
  std::clog << "-- wrote metadata file: " << join_path(tmp_dir, "metadata")
            << std::endl;

  write_vocab_to_file(vocab_txt_const, "vocab.txt", tmp_dir);
  auto const vocab_fn = join_path(tmp_dir, "vocab.txt");
  std::clog << "-- wrote vocabulary file: " << vocab_fn << std::endl;

  write_sample_list_to_file(sample_list, "sample_list", tmp_dir);
  std::clog << "-- wrote sample_list file: "
            << join_path(tmp_dir, "sample_list") << std::endl;
}
