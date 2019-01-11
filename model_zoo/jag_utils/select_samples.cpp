#include <algorithm>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include "lbann/lbann.hpp"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "lbann/lbann.hpp"

using namespace std;
using namespace lbann;

#undef SANITY
#define SANITY

int main(int argc, char **argv) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  options *opts = options::get();
  opts->init(argc, argv);

  std::stringstream err;

  // sanity check the cmd line
  if (! (opts->has_string("index_fn") && opts->has_string("sample_mapping_fn") 
         && opts->has_int("num_samples") && opts->has_int("random_seed")
         && opts->has_string("output_fn"))) {
    if (master) {
      LBANN_ERROR("usage: select_samples --index_fn=<string> --sample_mapping_fn=<string> --num_samples=<int> --output_fn=<string> --random_seed=<int>\n\n");
    }
    exit(9);
  }

  const std::string index_fn = opts->get_string("index_fn");
  const std::string mapping_fn = opts->get_string("sample_mapping_fn");
  const std::string output_fn = opts->get_string("output_fn");
  size_t num_samples = opts->get_int("num_samples");
  int seed = opts->get_int("random_seed");

  //==========================================================================
  // read previously computed mapping: sample_id (string) -> local_index
  //==========================================================================
  cerr << "reading sample mapping\n";
  // maps filename to { sample_ids }
  unordered_map<string, std::unordered_set<string>> sample_mapping;
  unordered_map<string, std::vector<string>> sample_mapping_v;
  // maps a sampleID to a local idex
  unordered_map<string, int> string_to_index;

  ifstream in(mapping_fn.c_str());
  string filename;
  string sample_id;
  string line;
  size_t n = 0;
  while (getline(in, line)) {
    if (!line.size()) {
      break;
    }
    stringstream s(line);
    s >> filename;
    ++n;
    int hh = 0;
    while (s >> sample_id) {
      sample_mapping[filename].insert(sample_id);
      sample_mapping_v[filename].push_back(sample_id);
      if (string_to_index.find(sample_id) != string_to_index.end()) {
        err << "duplicate sample_ID: " << sample_id << " in file: " << filename;
        LBANN_ERROR(err.str());
      }
      string_to_index[sample_id] = hh++;
    }
  }
  in.close();
  cerr << "num lines processed: " << n << "\n";

  //==========================================================================
  // master builds two maps: <string, set<int>> maps a filename to the
  // set of indices (not sample_ids; that comes later!) that are to be 
  // included and excluded
  if (master) {

    // your job, should you decide to accept it, is to fill in these maps
    std::unordered_map<std::string, std::unordered_set<int>> index_map_keep;
    std::unordered_map<std::string, std::unordered_set<int>> index_map_exclude;

    //open input file
    in.open(index_fn);
    if (!in) {
      err << "failed to open " << index_fn << " for reading\n";
      LBANN_ERROR(err.str());
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    getline(in, line);
    if (line != "CONDUIT_HDF5_EXCLUSION") {
      LBANN_ERROR("error: 1st line in index file must contain: CONDUIT_HDF5_EXCLUSION\n");
    }

    int num_valid, num_invalid, num_files;
    in >> num_valid >> num_invalid >> num_files;
    getline(in, line);  //discard newline 
    string base_dir;
    getline(in, base_dir);
    cerr << "input index file contains " << num_valid << " valid samples\n";

    // generate random indices; note that these are global indices
    cerr << "generating random indicess ...\n";
    unordered_set<int> random_indices;
    srandom(seed);
    while (true) {
      int v = random() % num_valid;
      random_indices.insert(v);
      if (random_indices.size() == num_samples) {
        break;
      }
    }
  
    // loop over each entry from in input index file; determine which, if any,
    // local indices will be added to the INCLUSION index
    int first = 0;
    size_t good, bad;
    num_files = 0;
    string fn;
    std::unordered_map<std::string, std::string> data;
    while (! in.eof()) {
      line = "";
      getline(in, line);
      if (!line.size()) {
        break;
      }
      ++num_files;
      if (num_files % 1000 == 0) cerr << num_files/1000 << "K input lines processed\n";
      stringstream s(line);
      s >> fn >> good >> bad;
      data[fn] = line;
      const int total = good+bad;
      index_map_exclude[fn];
      index_map_keep[fn];
      while (s >> sample_id) {
        if (sample_mapping[fn].find(sample_id) == sample_mapping[fn].end()) {
          LBANN_ERROR("failed to find " + sample_id + " in sample_mapping");
        }
        index_map_exclude[fn].insert(string_to_index[sample_id]);
      }
      if (index_map_exclude[fn].size() != bad) {
        err << "exclude.size(): " << index_map_exclude[fn].size() << " should be: " << bad << " but isn't\n";
        LBANN_ERROR(err.str());
      }

      int local_valid_index = 0;
      for (int local_index=0; local_index<total; local_index++) {
        if (index_map_exclude[fn].find(local_index) == index_map_exclude[fn].end()) {
          int global_idx = local_valid_index+first;
          ++local_valid_index;
          if (random_indices.find(global_idx) != random_indices.end()) {
            index_map_keep[fn].insert(local_index);
            index_map_exclude[fn].insert(local_index);
          }
        }
      }
      first += good;
    }

    if (index_map_exclude.size() != index_map_keep.size()) {
      LBANN_ERROR("index_map_exclude.size() != index_map_keep.size()");
    }

    //=====================================================================
    // write EXCLUSION file
    //=====================================================================
    //open output file and write 1st header line
    const std::string name1 = output_fn + "_bar";
    std::cerr << "\nWRITING output file: " << name1 << "\n";
    std::ofstream out(name1.c_str());
    if (!out) {
      err << "failed to open " << name1 << " for writing\n";
      LBANN_ERROR(err.str());
    }
    out<< "CONDUIT_HDF5_EXCLUSION\n";

    std::stringstream sout;
    size_t total_good = 0;
    size_t total_bad = 0;
    size_t num_include_files = 0;

    for (auto t : index_map_exclude) {
      filename = t.first;
      if (data.find(filename) == data.end()) {
        err << "data.find(" << filename << ") failed\n";
        for (auto tt : data) {
          err << tt.first << "\n";
        }
        LBANN_ERROR(err.str());
      }

      // get total samples for the current file
      std::stringstream s5(data[filename]);
      s5 >> fn >> good >> bad;
      size_t total = good+bad;

      const std::unordered_set<int> &exclude_me = t.second;
      int excluded = exclude_me.size();
      int included = total - excluded;
      if (included) {
        ++num_include_files;
        total_good += included;
        total_bad += excluded;
        sout << filename << " " << included << " " << excluded;
        for (auto t3 : exclude_me) {
          sout << " " << sample_mapping_v[fn][t3];
        }
        sout << "\n";
      }
    }

    out << total_good << " " << total_bad << " " << num_include_files << "\n"
        << base_dir << "\n" << sout.str();
    out.close();

    //=====================================================================
    // write INCLUSION file
    //=====================================================================
    // open output file and write 1st header line
    out.open(output_fn.c_str());
    std::cerr << "\nWRITING output file: " << output_fn << "\n";
    if (!out) {
      err << "failed to open " << output_fn << " for writing\n";
      LBANN_ERROR(err.str());
    }
    out << "CONDUIT_HDF5_INCLUSION\n";

    sout.clear();
    sout.str("");
    total_good = 0;
    total_bad = 0;
    num_include_files = 0;

    for (auto t : index_map_keep) {
      filename = t.first;
      if (data.find(filename) == data.end()) {
        err << "data.find(" << filename << ") failed\n";
        for (auto tt : data) {
          err << tt.first << "\n";
        }
        LBANN_ERROR(err.str());
      }

      // get total samples for the current file
      std::stringstream s5(data[filename]);
      s5 >> fn >> good >> bad;
      size_t total = good+bad;
      const std::unordered_set<int> &include_me = t.second;
      int included = include_me.size();
      int excluded = total - included;

      if (included) {
        ++num_include_files;
        total_good += included;
        total_bad += excluded;
        sout << filename << " " << included << " " << excluded;
        for (auto t3 : include_me) {
          sout << " " << sample_mapping_v[fn][t3];
        }
        sout << "\n";
      }
    }

    out << total_good << " " << total_bad << " " << num_include_files 
            << "\n" << base_dir << "\n" << sout.str();
  }


  finalize(comm);
}
