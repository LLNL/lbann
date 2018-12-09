////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include <time.h>

using namespace lbann;

void get_input_names(std::unordered_set<std::string> &s);
void get_scalar_names(std::unordered_set<std::string> &s);
//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    // sanity check invocation
    if (!opts->has_string("filelist")) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string>");
      }
    }

    // master reads the filelist and bcasts to others
    std::vector<std::string> files;
    std::string f;
    int size;
    if (master) {
      std::stringstream s;
      std::ifstream in(opts->get_string("filelist").c_str());
      if (!in) {
          throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + opts->get_string("filelist") + " for reading");
      }
      std::string line;
      while (getline(in, line)) {
        if (line.size()) {
          s << line << " ";
          //files.push_back(line);
        }
      }
      in.close();
      f = s.str();
      size = s.str().size();
      std::cout << "size: " << size << "\n";
    }
    comm->world_broadcast<int>(0, &size, 1);
    f.resize(size);
    comm->world_broadcast<char>(0, &f[0], size);

    // unpack the filenames into a vector
    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        files.push_back(filename);
      }
    }
    if (rank == 1) std::cerr << "num files: " << files.size() << "\n";

    std::unordered_set<std::string> input_names;
    std::unordered_set<std::string> scalar_names;
    get_input_names(input_names);
    get_scalar_names(scalar_names);

    // detect corruption!
    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;
    size_t h = 0;
    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      //if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";

      try {

        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (std::exception e) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (std::exception e) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      }
      std::cerr << rank << " :: " << files[j] << " contains " << cnames.size() << " samples\n";

      for (size_t i=0; i<cnames.size(); i++) {

        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (std::exception e) {
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {
            try {
 
              for (auto t : input_names) {
                key = cnames[i] + "/inputs/" + t;
                conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              }  
            } catch (std::exception e) {
              std::cerr << rank << " :: " << "exception reading an input for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }
  
            try {
              for (auto t : scalar_names) {
                key = cnames[i] + "/outputs/scalars/" + t;
                conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              }  
            } catch (std::exception e) {
              std::cerr << rank << " :: " << "exception reading an scalar for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

            try {  
              key = cnames[i] + "/outputs/images/(0.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            } catch (std::exception e) {
              std::cerr << rank << " :: " << "exception reading image: (0.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

            try { 
              key = cnames[i] + "/outputs/images/(90.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            } catch (std::exception e) {
              std::cerr << rank << " :: " << "exception reading image: (90.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }
  
           
            try { 
              key = cnames[i] + "/outputs/images/(90.0, 78.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            } catch (std::exception e) {
              std::cerr << rank << " :: " << "exception reading image: (90.0, 78.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }
          }
        }
      }
  } catch (exception const &e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    finalize(comm);
    return EXIT_FAILURE;
  }

  // Clean up
  finalize(comm);
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
#endif //#ifdef LBANN_HAS_CONDUIT
