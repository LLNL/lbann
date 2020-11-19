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
    bool is_shared_training_data_reader = false;
    bool is_shared_testing_data_reader = false;

    init_data_readers(comm.get(), pb, data_readers, is_shared_training_data_reader, is_shared_testing_data_reader);

    // exercise a bit of the reader's API functionality
    for (std::map<execution_mode, generic_data_reader *>::iterator iter = data_readers.begin(); iter != data_readers.end(); iter++) {
      generic_data_reader *base_reader = iter->second;
      std::cerr << "main: calling preload_data_store()" << std::endl;
      base_reader->preload_data_store();
      std::cerr << "DONE! main: calling preload_data_store()" << std::endl;
      // hdf5_data_reader* hdf5
      // ...
      hdf5_data_reader *reader = dynamic_cast<hdf5_data_reader*>(base_reader);
      if (reader == nullptr) {
        LBANN_ERROR("reader == nullptr");
      }
      const std::unordered_set<std::string> &field_names = reader->get_field_names();
//XX
      for (auto t : field_names) {
        std::cout << "next field: " << t << std::endl;
      }

      for (auto pathname : field_names) {
        if (true) {
        //if (pathname.find("emi") == std::string::npos) {
//          size_t num_elts;
          //const conduit::char_array data = reader->get_data2(0, pathname, num_elts);
//          std::cout << "NUM_ELTS: " << num_elts << std::endl;
//        const conduit::char_array hdf5_data_reader::get_data2(const size_t sample_id, const std::string &field_name, size_t &num_elts) const {

        /*
          if (reader->is_double(pathname)) {
            double *xx;
            reader->get_data<double>(0, pathname, xx, n_elts);
//            std::cout << pathname << " " << *x << std::endl;
          } else if (reader->is_float(pathname)) {
            float *xxx;
            reader->get_data<float>(0, pathname, xxx, n_elts);
//            std::cout << pathname << " " << *x << std::endl;
          } else {
            LBANN_ERROR("not float or double!");
          }
          */
        }
      }
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
