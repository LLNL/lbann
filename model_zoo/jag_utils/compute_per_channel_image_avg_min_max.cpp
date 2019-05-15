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

#ifdef LBANN_HAS_CONDUIT

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include <time.h>
#include <cfloat>

using namespace lbann;

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

    if (!(opts->has_string("filelist"))) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string>");
      }
    }

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

    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        files.push_back(filename);
      }
    }
    if (rank==1) std::cerr << "num files: " << files.size() << "\n";

    //=======================================================================

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    //    if (master) std::cout << np << hdf5_file_hnd << "\n";

    int num_samples = 0;

    std::vector<float> v_max(4, FLT_MIN);
    std::vector<float> v_min(4, FLT_MAX);
    std::vector<double> v_sum(4, 0.0);
    std::vector<long>  v_num_pixels(4, 0);

    size_t h = 0;
    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";

      try {
std::cerr << rank << " :: opening for reading: " << files[j] << "\n";
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      }

      for (size_t i=0; i<cnames.size(); i++) {

        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {


            try {
              key = cnames[i] + "/outputs/images/(0.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              //out.write((char*)&emi[0], sizeof(float)*image_size);
              for (int channel = 0; channel<4; channel++) {
                for (size_t hh=channel; hh<image_size; hh += 4) {
                  float val = emi[hh];
                  if (val < v_min[channel]) v_min[channel] = val;
                  if (val > v_max[channel]) v_max[channel] = val;
                  v_sum[channel] += val;
                  v_num_pixels[channel]++;
                }
              }
            } catch (...) {
              std::cerr << rank << " :: " << "exception reading image: (0.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

            try {
              key = cnames[i] + "/outputs/images/(90.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              //out.write((char*)&emi[0], sizeof(float)*image_size);
              for (int channel = 0; channel<4; channel++) {
                for (size_t hh=channel; hh<image_size; hh += 4) {
                  float val = emi[hh];
                  if (val < v_min[channel]) v_min[channel] = val;
                  if (val > v_max[channel]) v_max[channel] = val;
                  v_sum[channel] += val;
                  v_num_pixels[channel]++;
                }
              }
            } catch (...) {
              std::cerr << rank << " :: " << "exception reading image: (90.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

            try {
              key = cnames[i] + "/outputs/images/(90.0, 78.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              //out.write((char*)&emi[0], sizeof(float)*image_size);
              for (int channel = 0; channel<4; channel++) {
                for (size_t hh=channel; hh<image_size; hh += 4) {
                  float val = emi[hh];
                  if (val < v_min[channel]) v_min[channel] = val;
                  if (val > v_max[channel]) v_max[channel] = val;
                  v_sum[channel] += val;
                  v_num_pixels[channel]++;
                }
              }
            } catch (...) {
              std::cerr << rank << " :: " << "exception reading image: (90.0, 78.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

          ++num_samples;
        }
      }
    }


    std::vector<float> global_v_min(4);
    std::vector<float> global_v_max(4);
    std::vector<double> global_v_sum(4);
    std::vector<long>  global_v_num_pixels(4);
    MPI_Reduce(v_min.data(), global_v_min.data(), 4, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(v_max.data(), global_v_max.data(), 4, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(v_sum.data(), global_v_sum.data(), 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(v_num_pixels.data(), global_v_num_pixels.data(), 4, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<double> global_v_avg(4);
    for (int j=0; j<4; j++) {
      global_v_avg[j] = global_v_sum[j] / global_v_num_pixels[j];
    }

    if (master) {
      for (int j=0; j<4; j++) {
        std::cout << global_v_min[j] << " " << global_v_max[j] << " " << global_v_avg[j] << " " << global_v_sum[j] << " " << global_v_num_pixels[j] << "\n";
      }
    }

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



#endif //#ifdef LBANN_HAS_CONDUIT
