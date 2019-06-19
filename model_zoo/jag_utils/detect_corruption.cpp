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

#include "lbann_config.hpp"

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include "lbann/utils/jag_utils.hpp"
#include <time.h>

using namespace lbann;

void get_input_names(std::unordered_set<std::string> &s);
void get_scalar_names(std::unordered_set<std::string> &s);
void get_image_names(std::unordered_set<std::string> &s);
void print_errs(world_comm_ptr &comm, int np, int rank, std::ostringstream &s, const char *msg);

//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    // sanity check invocation
    if (!opts->has_string("filelist")) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string> \nwhere: 'filelist' is a file that contains the fully qualified filenames of the conduit *'bundle' files that are to be inspected.\nfunction: attemptsto detect and report currupt files and/or samples within those files.");
      }
    }

    const std::string fn = opts->get_string("filelist");
    std::vector<std::string> filenames;
    read_filelist(comm.get(), fn, filenames);

    std::unordered_set<std::string> input_names;
    std::unordered_set<std::string> scalar_names;
    std::unordered_set<std::string> image_names;
    get_input_names(input_names);
    get_scalar_names(scalar_names);
    get_image_names(image_names);

    if (master) {
      std::cerr << "\nchecking the following inputs: \n";
      for (auto t : input_names) std::cerr << t << " ";
      std::cerr << "\n";
      std::cerr << "\nchecking the following scalars: ";
      for (auto t : scalar_names) std::cerr << t << " ";
      std::cerr << "\n";
      std::cerr << "\nchecking the following images: ";
      for (auto t : image_names) std::cerr << t << " ";
      std::cerr << "\n\n";
    }

    //================================================================
    // detect corruption!

    //these  error conditions ar liste in the order in which they're
    //tested. Upon failure, we call continue," i.e, no further tests
    //are cunducted
    std::ostringstream open_err;         //failed to open file
    std::ostringstream children_err;     //failed to read child names
    std::ostringstream success_flag_err; //failed to read success flag

    std::ostringstream sample_err; //catch all for errors in reading inputs,
                                  //scalars, and images
    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;
    int h = 0;

    // used to ensure all values are used
    double total = 0;
    for (size_t j=rank; j<filenames.size(); j+= np) {
      h += 1;
      if (h % 1 == 0 && master) std::cerr << "P_0 has processed " << h << " files\n";

      try {
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filenames[j].c_str() );
      } catch (...) {
        open_err << filenames[j] << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (...) {
        children_err << filenames[j] << "\n";
        continue;
      }

      for (size_t i=0; i<cnames.size(); i++) {
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          success_flag_err << filenames[j] << " " << cnames[i] << "\n";
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
            try {
              for (auto t : input_names) {
                key = cnames[i] + "/inputs/" + t;
                conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
                total += static_cast<double>(tmp.value());
              }
            } catch (...) {
              success_flag_err << filenames[j] << "\n";
              sample_err << filenames[j] << " " << cnames[i] << "\n";
              continue;
            }

            try {
              for (auto t : scalar_names) {
                key = cnames[i] + "/outputs/scalars/" + t;
                conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
                total += static_cast<double>(tmp.value());
              }
            } catch (...) {
              sample_err << filenames[j] << " " << cnames[i] << "\n";
              continue;
            }

            try {
              for (auto t : image_names) {
                key = cnames[i] + "/outputs/images/" + t + "/0.0/emi";
                conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
                conduit::float32_array emi = tmp.value();
                const size_t image_size = emi.number_of_elements();
                for (size_t k=0; k<image_size; k++) {
                  total += emi[k];
                }
              }
            } catch (...) {
              sample_err << filenames[j] << " " << cnames[i] << "\n";
              continue;
            }
          }
        }
      }

      if (master) {
        int h2 = comm->reduce<int>(h, comm->get_world_comm());
        double total2 = comm->reduce<double>(total, comm->get_world_comm());
        std::cerr << "\nnum files processed: " << h2 << "\n"
                  << "sanity check - please ignore: " << total2 << "\n\n";
      } else {
        comm->reduce<int>(h, 0, comm->get_world_comm());
        comm->reduce<double>(total, 0, comm->get_world_comm());
      }

      // print erros, if any
      print_errs(comm, np, rank, open_err, "failed to open these files (if any):");
      print_errs(comm, np, rank, children_err, "failed to read children from these files (if any):");
      print_errs(comm, np, rank, success_flag_err, "failed to read success flag for these samples (if any):");
      print_errs(comm, np, rank, sample_err, "failed to read input or scalars or images for these samples (if any):");

  } catch (exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}

void get_input_names(std::unordered_set<std::string> &s) {
  s.insert("shape_model_initial_modes:(4,3)");
  s.insert("betti_prl15_trans_u");
  s.insert("betti_prl15_trans_v");
  s.insert("shape_model_initial_modes:(2,1)");
  s.insert("shape_model_initial_modes:(1,0)");
}

void get_scalar_names(std::unordered_set<std::string> &s) {
  s.insert("BWx");
  s.insert("BT");
  s.insert("tMAXt");
  s.insert("BWn");
  s.insert("MAXpressure");
  s.insert("BAte");
  s.insert("MAXtion");
  s.insert("tMAXpressure");
  s.insert("BAt");
  s.insert("Yn");
  s.insert("Ye");
  s.insert("Yx");
  s.insert("tMAXte");
  s.insert("BAtion");
  s.insert("MAXte");
  s.insert("tMAXtion");
  s.insert("BTx");
  s.insert("MAXt");
  s.insert("BTn");
  s.insert("BApressure");
  s.insert("tMINradius");
  s.insert("MINradius");
}

void get_image_names(std::unordered_set<std::string> &s) {
  s.insert("(0.0, 0.0)");
  s.insert("(90.0, 0.0)");
  s.insert("(90.0, 78.0)");
}

void print_errs(world_comm_ptr &comm, int np, int rank, std::ostringstream &s, const char *msg) {
  comm->global_barrier();
  if (rank == 0) { std::cerr << "\n" << msg << "\n"; }
  for (int i=0; i<np; i++) {
    comm->global_barrier();
    if (rank == i) {
        std::cerr << s.str();
    }
  }
  comm->global_barrier();
}
