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

//#include <vector>
#include "lbann/comm.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/commify.hpp"
#include <cnpy.h>
#include <cmath>

using namespace lbann;

#if 0
struct xyz {
  xyz(float xx, float yy, float zz) : x(xx), y(yy), z(zz) { }

  float x;
  float y;
  float z;

  float dist(const xyz &point) {
    return sqrt( 
             (pow( (x-xyz.x), 2) 
             + pow( (x-xyz.x), 2) 
             + pow( (x-xyz.x), 2))
           );  
  }
};
#endif

int main(int argc, char *argv[]) {
#if 0
  int random_seed = 0;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();

  try {
    options *opts = options::get();
    opts->init(argc, argv);

    if (! opts->has_string("filelist")) {
      LBANN_ERROR("usage: ", argv[0], " --filelist=<string>");
    }

    const std::string input_fn = opts->get_string("filelist");

    int rank = comm->get_rank_in_world();
    int np = comm->get_procs_in_world();

    // get list of input filenames
    std::vector<std::string> filenames;
    read_filelist(comm.get(), input_fn, filenames);

    const std::map<std::string, cnpy::NpyArray> aa = cnpy::npz_load(filenames[ran]);
    const std::vector<size_t> shape = aa["bbs"].shape;
    const size_t              word_size = aa["bbs"].word_size;
    const size_t              num_vals = aa["bbs"].num_vals;

std::cout << "shape: ";
for (auto t : shape) std::cout << t<< " ";
std::cout << "\nword_size: " << word_size << "\nnum_vals: " <<num_vals
          << std::endl;
exit(0);

    size_t nn = 0; // only for user feedback
    for (size_t j=rank; j<filenames.size(); j+=np) {
      if (master) {
        std::cerr << "master is loading: " << filenames[j] << std::endl;
      }

      size_t jj = filenames[j].rfind(".");
      if (jj == std::string::npos) {
        LBANN_ERROR("failed find '.' in filename: ", filenames[j]);
      }
      const std::string fn = filenames[j].substr(0, j) + ".bbs_stat";
      std::ofstream out(fn);
      if (!out) {
        LBANN_ERROR("failed to open ", fn, " for writing");
      }

std::cout << "will open: " << fn << " for writing" << std::endl;
exit(0);

      std::map<std::string, cnpy::NpyArray> a = cnpy::npz_load(filenames[j]);
      ++nn;


      out.close();
      const float *data = a[name].data();
      size_t num_bytes = a[name].num_bytes();

/*
      size_t n_elts = a["density_sig1"].num_vals;
      double *data = reinterpret_cast<double*>(a["density_sig1"].data_holder->data());
*/

      if (master) {
        std::cerr << "approx " << utils::commify(nn*np) << " files of " 
                  << filenames.size() << " processed\n";
      }
    }
    // ==================== finished processing all files ========================


  } catch (std::exception const &e) {
    if (master) std::cerr << "caught exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "unknown exception in main\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
#endif
}

