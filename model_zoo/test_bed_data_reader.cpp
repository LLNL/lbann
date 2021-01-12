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
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf_utils.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/utils/argument_parser.hpp"

#include <lbann.pb.h>
#include <model.pb.h>

#include <cstdlib>

using namespace lbann;

int main(int argc, char *argv[]) {
  auto& arg_parser = global_argument_parser();
  construct_std_options();
  try {
    arg_parser.parse(argc, argv);
  }
  catch (std::exception const& e) {
    std::cerr << "Error during argument parsing:\n\ne.what():\n\n  "
              << e.what() << "\n\nProcess terminating."
              << std::endl;
    std::terminate();
  }

  world_comm_ptr comm = initialize(argc, argv);
  const bool master = comm->am_world_master();

  int random_seed = 10;
  int io_threads_per_process = 1;
  init_random(random_seed, io_threads_per_process);
  init_data_seq_random(random_seed+1); // is this needed?

  try {
    options *opts = options::get();
    opts->init(argc, argv);
    auto pbs = protobuf_utils::load_prototext(master, argc, argv);
    for(size_t i = 0; i < pbs.size(); i++) {
      get_cmdline_overrides(*comm, *(pbs[i]));
    }
    lbann_data::LbannPB& pb = *(pbs[0]);

    std::map<execution_mode, generic_data_reader *> data_readers;
    init_data_readers(comm.get(), pb, data_readers);

    // exercise a bit of the reader's API functionality
    for (std::map<execution_mode, generic_data_reader *>::iterator iter = data_readers.begin(); iter != data_readers.end(); iter++) {
      generic_data_reader *base_reader = iter->second;
      base_reader->preload_data_store();
      if (master) {
        std::cout << "reader with role " << base_reader->get_role() 
                  << " has " << base_reader->get_num_data() << " samples\n";
      }
      smiles_data_reader* reader = dynamic_cast<smiles_data_reader*>(base_reader);
      if (reader == nullptr) {
        LBANN_ERROR("smiles_data_reader == nullptr");
      }
      std::set<int> indices = reader->get_my_indices();
      std::cout << "num my indices: " << indices.size() << std::endl;

      int passed = 0;
      int failed = 0;

      // this mimics a call to smiles_data_reader::fetch_datum; in fact,
      // this is where the data is retrieved; the remainder of fetch_datum
      // just stuffs it into a matrix
      for (auto index : indices) {
        const conduit::Node& node = reader->get_data_store_ptr()->get_conduit_node(index);
        const conduit::unsigned_short_array tmp_data = node["/" + LBANN_DATA_ID_STR(index) + "/data"].as_unsigned_short_array();
        std::vector<unsigned short> data;
        size_t n = tmp_data.number_of_elements();
        for (size_t j=0; j<n; j++) {
          data.push_back(tmp_data[j]);
        }

        //decode data back to a string
        std::string decoded;
        reader->decode_smiles(data, decoded);

        //read the smiles string directly from file
        std::string filename;
        size_t offset;
        short length;
        reader->get_sample_origin(index, filename, offset, length);
        std::ifstream in(filename.c_str());
        if (!in) {
          LBANN_ERROR("failed to open ", filename, " for reading");
        }
        in.seekg(offset);
        std::string from_file;
        from_file.resize(length);
        in.read((char*)from_file.data(), length);
        in.close();

        std::vector<unsigned short> f_test;
        reader->encode_smiles(from_file, f_test);
        std::string f_test_str;
        reader->decode_smiles(f_test, f_test_str);

        if (decoded != from_file) {
          ++failed;
          std::cerr << "Failed: " << index << "\n"
                    << "  "<<decoded<<" //decoded"<<std::endl
                    << "  "<<from_file<<" //from_file"<<std::endl
                    << "  "<< f_test_str<<"//from file, encoded+decoded" << std::endl;
          //LBANN_ERROR("decoded: ", decoded, " != from_file: ", from_file);
        } else {
          ++passed;
          std::cout << "passed: " << index << " :: " << decoded << std::endl;
        }  
      }
      std::cout << std::endl << "num passed: " << passed << std::endl;
      std::cout << std::endl << "num failed: " << failed << std::endl;
    }

  } catch (exception& e) {
    El::ReportException(e);
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  } catch (std::exception& e) {
    El::ReportException(e);
    El::mpi::Abort(El::mpi::COMM_WORLD, EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
