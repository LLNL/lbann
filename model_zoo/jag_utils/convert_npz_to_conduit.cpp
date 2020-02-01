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

#include <vector>
#include "lbann/comm.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/commify.hpp"
#include <cnpy.h>
#include <numeric>
#include "conduit/conduit_node.hpp"
#include "lbann/data_store/data_store_conduit.hpp" //LBANN_DATA_ID_STR
#include "conduit/conduit_relay_io_hdf5.hpp"


using namespace lbann;

int main(int argc, char *argv[]) {
  int random_seed = 0;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
    // Initialize options db (this parses the command line)
    options *opts = options::get();
    opts->init(argc, argv);

    if (! opts->has_string("filelist")) {
      if (master) {
        std::cerr << "usage: " << argv[1] << " --filelist=<string>\n"
                  << "function: converts npz files to conduit\n";
      }
      comm->global_barrier();
      return EXIT_FAILURE;
    }

    const std::string input_fn = opts->get_string("filelist");

    int rank = comm->get_rank_in_world();
    int np = comm->get_procs_in_world();

    // get list of input filenames
    std::vector<std::string> filenames;
    read_filelist(comm.get(), input_fn, filenames);


    // get the shape vectors; note that shape[0] is the number of
    // samples (aka, frames)
    std::map<std::string, cnpy::NpyArray> aaa = cnpy::npz_load(filenames[0]);
    std::unordered_map<std::string, std::vector<size_t>> shapes;
    for (const auto &t : aaa) {
      const std::vector<size_t> &shape = t.second.shape;
      if (shape.size() == 1) {
        shapes[t.first].push_back(1);
      } else {
        for (size_t x=1; x<shape.size(); x++) {
          shapes[t.first].push_back(shape[x]);
        }
      }
    }

    // get the number of elements in each field
    std::unordered_map<std::string, size_t> num_words;
    for (const auto &t : shapes) {
      num_words[t.first] = std::accumulate(t.second.begin(), t.second.end(), 1, std::multiplies<int>());
    }

    for (size_t j=rank; j<filenames.size(); j+=np) {
      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      conduit::Node node;
      int num_samples = a["frames"].shape[0];
      for (int sample_index = 0; sample_index < num_samples; sample_index++) {
        for (const auto &t : a) {
          const std::string &name = t.first;

          if (name == "frames") {
            //pass
          } 

          else if (name == "bbs") {
            float *data = a[name].data<float>();
            size_t offset = sample_index*num_words["bbs"];
            node[LBANN_DATA_ID_STR(sample_index) + "/" + name + "/data"].set(data + offset, num_words[name]);
            node[LBANN_DATA_ID_STR(sample_index) + "/" + name + "/shape"].set(shapes[name]);
            node[LBANN_DATA_ID_STR(sample_index) + "/" + name + "/size"].set(num_words[name]);
          } 

          else { // rots, states, tilts, density_sig1, probs
            size_t offset = sample_index*num_words[name];
            double *data = a[name].data<double>();
            node[LBANN_DATA_ID_STR(sample_index) + "/" + name + "/data"].set(data + offset, num_words[name]);
            node[LBANN_DATA_ID_STR(sample_index) + "/" + name + "/shape"].set(shapes[name]);
            node[LBANN_DATA_ID_STR(sample_index) + "/" + name + "/size"].set(num_words[name]);
          }

        }
      }

      // save to file
      size_t n2 = filenames[j].rfind(".");
      if (n2 == std::string::npos) {
        LBANN_ERROR("n2 == std::string::npos");
      }
      std::string fn2 = filenames[j].substr(0, n2) + ".bin";
      std::string fn3 = filenames[j].substr(0, n2) + ".hdf5";
      node.save(fn2);
      conduit::relay::io::hdf5_save(node, fn3);
    }

  } catch (std::exception const &e) {
    if (master) std::cerr << "caught exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}

