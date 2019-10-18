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
#include <time.h>

using namespace lbann;

#define NUM_OUTPUT_DIRS 100
#define NUM_SAMPLES_PER_FILE 1000

//==========================================================================
void check_invocation(bool master);

std::string usage();

int construct_output_directories(int np);

// Compute set of random (global) sample indices;
void get_random_sample_indices(
  const std::unordered_set<int> &exclude,
  std::set<int> &indices,
  int global_num_samples);

// Construct set of sample_IDs that should not appear in the output;
// the intent is that, if random sample(s) were previously extracted,
// we don't want any overlap
void build_exclusion_set(std::unordered_set<int> &exclude);

void get_global_num_samples(int &num_samples, int&num_files);

void build_sample_mapping(
  std::vector<std::string> &filenames,
  const std::set<int> &indices,
  std::vector<std::set<int> > &samples);

void extract_samples(
  lbann_comm *comm,
  const int rank,
  const int np,
  const std::vector<std::string> &filenames,
  const std::vector<std::set<int> > &samples);

// debug function
void print_sample_ids(
  const std::vector<std::string> &filenames,
  const std::vector<std::set<int> > &samples);
//==========================================================================
int main(int argc, char *argv[]) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {

    // Optionally print usage instructions then exit
    if (argc == 1) {
      if (master) {
        std::cout << usage();
      }
      return EXIT_SUCCESS;
    }

    options *opts = options::get();
    opts->init(argc, argv);

    // ensure the db contains "num_samples_per_file"
    opts->get_int("num_samples_per_file", NUM_SAMPLES_PER_FILE);

    int num_output_dirs;
    if (master) {
      check_invocation(master);
      num_output_dirs = construct_output_directories(np);
    }
    comm->world_broadcast<int>(0, &num_output_dirs, 1);
    opts->set_option("num_output_dirs", num_output_dirs);

    // get the set of global indices for the samples in our extracted set
    std::set<int> indices;
    std::unordered_set<int> exclude;
    int global_num_samples;
    int num_files;
    get_global_num_samples(global_num_samples, num_files);

    build_exclusion_set(exclude);
    std::vector<int> indices_v;
    size_t num_samples = options::get()->get_int("num_samples");
    indices_v.reserve(num_samples);

    get_random_sample_indices(exclude, indices, global_num_samples);

    if (master) {
      // write set of random indices to file; these can be used
      // as an exclusion set for a subsequent run
      const std::string base_dir = opts->get_string("output_base_dir");
      const std::string fn = base_dir + "/random_indices.txt";
      std::ofstream out(fn.c_str());
      if (!out) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + base_dir + "/random_indices.txt for writing");
      }
      for (auto t : indices) {
        out << t << "\n";
        indices_v.push_back(t);
      }
    }

    // samples[j] contains local indices for samples wrt the j-th conduit file
    std::vector<std::set<int> > samples;
    std::vector<std::string> conduit_filenames;
    build_sample_mapping(conduit_filenames, indices, samples);
    num_files = samples.size();

    extract_samples(comm.get(), rank, np, conduit_filenames, samples);

  } catch (const exception& e) {
    std::cerr << "\n\n" << rank << " ::::: caught exception, outer try/catch: " << e.what() << "\n\n";
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}

void get_random_sample_indices(const std::unordered_set<int> &exclude, std::set<int> &indices, int global_num_samples) {
  size_t num_samples = options::get()->get_int("num_samples");
  int seed = options::get()->get_int("rand_seed");
  srand(seed);
  while (indices.size() < num_samples) {
    int v = rand() % global_num_samples;
    if (indices.find(v) == indices.end() && exclude.find(v) == exclude.end()) {
      indices.insert(v);
    }
  }
}

std::string usage() {
    std::string u =
      "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
      "usage: extract_random_samples --index_fn=<string> --num_samples=<int> --output_base_dir=<string> --rand_seed=<int> [ --exclude=<string> ] [ --num_samples_per_output_file=<int> ]\n"
      "where: --index_fn is the output file from the build_index executable\n"
      "       --num_samples is the number of random samples to be extracted\n"
      "       --output_base_dir will be created if it doesn't exist\n"
      "       --exclude is an optional filename containing IDs of samples that should not appear in the output\n"
      "       --rand_seed is required to ensure all procs generate identical random sample indices.\n"
      "       --num_samples_per_file is number of samples per output file; default is 1000 (a maximum of one output file per processor may contain fewer)\n"
      "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n";
    return u;
}

void build_exclusion_set(std::unordered_set<int> &exclude) {
  const std::string exclude_fn = options::get()->get_string("exclude", "");
  if (exclude_fn == "") {
    return;
  }
  std::ifstream in(exclude_fn);
  if (!in) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + exclude_fn + " for reading");
  }
  int id;
  while (in >> id) {
    exclude.insert(id);
  }
  in.close();
}

int construct_output_directories(int np) {
  options *opts = options::get();
  int num_samples_per_file = opts->get_int("num_samples_per_file");
  int num_samples = opts->get_int("num_samples");
  const std::string base_dir = options::get()->get_string("output_base_dir");
  int num_output_dirs = ((num_samples / num_samples_per_file + 1) / np) *2;

  for (int j=0; j<num_output_dirs; j++) {
    std::stringstream s1;
    s1 << "mkdir -p " << base_dir << "/" << j;
    int err = system(s1.str().c_str());
    if (err != 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s1.str());
    }
  }
  return num_output_dirs;
}

void build_sample_mapping(
  std::vector<std::string> &filenames,
  const std::set<int> &indices,
  std::vector<std::set<int> > &samples) {

  // open index file
  const std::string index_fn = options::get()->get_string("index_fn");
  std::ifstream in(index_fn);
  if (!in) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + index_fn + " for reading");
  }

  // get num global samples, num files, and base directory for input
  // conduit files
  int n_files;
  int global_num_samples;
  std::string input_base_dir;
  in >> global_num_samples >> n_files >> input_base_dir;

  std::string entry;
  getline(in, entry); //discard newline

  int num_valid_samples;
  std::string conduit_fn;
  int start = 0;
  int end;
  while (!in.eof()) {
    getline(in, entry);
    if (entry.size()) {
      std::stringstream s(entry);
      s >> conduit_fn >> num_valid_samples;
      end = start + num_valid_samples;
      bool found = false;
      for (int j=start; j<end; j++) {
        if (indices.find(j) != indices.end()) {
          if (!found) {
            samples.resize(samples.size()+1);
	    const std::string f = input_base_dir + "/" + conduit_fn;
	    filenames.push_back(f);
            found = true;
          }
          samples.back().insert(j-start);
        }
      }
    }
    start += num_valid_samples;
  }
}

void check_invocation(bool master) {
  options *opts = options::get();
  if (! (opts->has_string("index_fn") && opts->has_int("num_samples") && opts->has_string("output_base_dir") && opts->has_int("rand_seed"))) {
    if (master) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: improper invocation; see usage message below\n\n " + usage() + "\n\n");
    }
  }
}

void get_global_num_samples(int &num_samples, int &num_files) {
  const std::string index_fn = options::get()->get_string("index_fn");
  std::ifstream in(index_fn);
  if (!in) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + index_fn + " for reading");
  }
  in >> num_samples >> num_files;
  in.close();
}

void extract_samples(
  lbann_comm *comm,
  const int rank,
  const int np,
  const std::vector<std::string> &filenames,
  const std::vector<std::set<int> > &samples) {

  const std::string base_dir = options::get()->get_string("output_base_dir");
  char b[1024];
  sprintf(b, "%s/_sample_ids_%d.txt", base_dir.c_str(), rank);
  std::ofstream out_ids(b);
  if (!out_ids) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: failed to open " + b + " for writing");
  }
  int num_output_dirs = options::get()->get_int("num_output_dirs");
  int num_samples_per_file = options::get()->get_int("num_samples_per_file");
  int file_id = 0;
  int dir_id = 0;
  int n_samples = 0;

  hid_t hdf5_file_hnd;
  conduit::Node n_ok;
  conduit::Node save_me;
  conduit::Node tmp;
  conduit::Node tmp2;
  std::string key;

std::cerr << rank << " samples.size: " << samples.size() << " np: " << np << "\n";

  size_t num_processed = 0;
  for (size_t j=rank; j<samples.size(); j+=np) {
    // open input conduit file
    try {
      hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filenames[j].c_str() );
    } catch (...) {
      std::cerr << rank << " :: exception hdf5_open_file_for_read; j: " << j <<  " filenames.size(): " << filenames.size() << " samples.size(): " << samples.size() << "\n";
      continue;
    }
    std::cerr << rank << " :: opened: " << filenames[j] << "\n";

    out_ids << filenames[j] << " ";

    std::vector<std::string> cnames;
    try {
      conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    } catch (...) {
      std::cerr << rank << " :: exception hdf5_group_list_child_names; " << filenames[j] << "\n";
      continue;
    }

    int local_idx = 0;
    for (size_t i=0; i<cnames.size(); i++) {
      // is the next sample valid?
      key = "/" + cnames[i] + "/performance/success";
      try {
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
      } catch (std::exception const &e) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: caught exception reading success flag for child " + std::to_string(i) + " of " + std::to_string(cnames.size()) + "; " + filenames[j] +  "\n");
      }
      int success = n_ok.to_int64();

      // if valid, perform the extraction
      if (success == 1) {
        if (samples[j].find(local_idx) != samples[j].end()) {
          ++n_samples;
          tmp2["/performance/success"] = 1;
          out_ids << cnames[i] << " ";
          try {
            key = cnames[i] + "/inputs";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            tmp2["/inputs"] = tmp;

            key = cnames[i] + "/outputs/scalars";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            tmp2["/outputs/scalars"] = tmp;

            key = cnames[i] + "/outputs/images/(0.0, 0.0)//0.0/emi";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            tmp2["/outputs/images/(0.0, 0.0)//0.0/emi"] = tmp;
            //save_me[cnames[i]]["/outputs/images"] = tmp;

            key = cnames[i] + "/outputs/images/(90.0, 0.0)//0.0/emi";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            tmp2["/outputs/images/(90.0, 0.0)//0.0/emi"] = tmp;
            //save_me[cnames[i]]["/outputs/images"] = tmp;

            key = cnames[i] + "/outputs/images/(90.0, 78.0)//0.0/emi";
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
            //save_me[cnames[i]]["/outputs/images"] = tmp;
            tmp2["/outputs/images/(90.0, 78.0)//0.0/emi"] = tmp;

          } catch (...) {
            std::cerr << rank << " :: " << "exception caught during extraction; ignoring and continuing\n";
            continue;
          }
          save_me[cnames[i]] = tmp2;

          if (n_samples >= num_samples_per_file) {
            std::stringstream s;
            if (dir_id == num_output_dirs) {
              dir_id = 0;
            }
            std::stringstream fn;
            fn << base_dir << "/" << dir_id++ << "/samples_" << rank
               << "_" << file_id++ << ".bundle";
            std::cerr << rank << " :: writing " << fn.str() << " file with " << n_samples << " samples\n";
            n_samples = 0;

            try {
              conduit::relay::io::save(save_me, fn.str(), "hdf5");
            } catch (...) {
              throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: exception conduit::relay::save()\n");
            }
            save_me.reset();
          }
        }
        ++local_idx;
      }
    }

    out_ids << "\n";

    try {
      conduit::relay::io::hdf5_close_file( hdf5_file_hnd );
    } catch (const exception& e) {
       throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: exception hdf5_close_file; " + filenames[j] + "; " + e.what());
    }

    ++num_processed;
    if (num_processed % 10 == 0) {
      std::cerr << rank << " :: " << num_processed << " files processed; n_samples: " << n_samples << "\n";
    }

  }

  // write final file
  if (n_samples) {
      std::cerr << "writing FINAL conduit file with " << n_samples << " samples\n";
    std::stringstream fn;
    fn << base_dir << "/" << dir_id++ << "/samples_" << rank
              << "_" << file_id++ << ".bundle";
    try {
      conduit::relay::io::save(save_me, fn.str(), "hdf5");
    } catch (const exception& e) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: exception conduit::relay::save(); what: " + e.what());
    }
  }

  out_ids.close();
  comm->global_barrier();
  if (!rank) {
    sprintf(b, "cat %s/_sample_id* > %s/sample_ids.txt", base_dir.c_str(), base_dir.c_str());
    int r = system(b);
    if (r != 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + b);
    }
    sprintf(b, "rm -f %s/_sample_id*", base_dir.c_str());
    r = system(b);
    if (r != 0) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + b);
    }
  }
}

void print_sample_ids(
  const std::vector<std::string> &filenames,
  const std::vector<std::set<int> > &samples) {

  std::cerr << "filenames size: " << filenames.size() << "\n";
  std::cerr << "samples size: " << samples.size() << "\n";
  for (size_t i=0; i<samples.size(); i++) {
    std::cerr << "i: " << i << " num samples: " << samples[i].size() << "\n";
  }

  std::cerr << "==========================================\n";
  std::cerr << "sample map:\n";
  for (size_t i=0; i<samples.size(); i++) {
    std::cout << filenames[i] << "\n";
    for (auto t : samples[i]) {
      std::cout << t << " ";
    }
    std::cout << "\n";
  }
  std::cerr << "\n==========================================\n";
}
