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

//============================================================================
// sanity check the cmd line
void check_cmd_line();

// returns the help message
string help_msg();

// tests that there are sufficient samples to build the lists
void sanity_test_request();

void read_mapping_file(
  unordered_map<string, unordered_set<string>> &sample_mapping, 
  unordered_map<string, vector<string>> &sample_mapping_v, 
  unordered_map<string, int>& string_to_index);

void build_index_maps(
  unordered_map<string, unordered_set<string>> &sample_mapping, 
  unordered_map<string, unordered_set<int>> &index_map_keep, 
  unordered_map<string, unordered_set<int>> &index_map_exclude,
  unordered_map<string, int> &string_to_index,
  unordered_map<string, string> &filename_data);

void divide_selected_samples(
  const unordered_map<string, unordered_set<int>> &index_map_keep,
  vector<unordered_map<string, unordered_set<int>>> &sets); 

//todo: some of these should be const
void write_sample_list(
    int n, 
    vector<unordered_map<string, unordered_set<int>>> &subsets, 
    unordered_map<string, vector<string>> &sample_mapping_v,
    std::unordered_map<std::string, std::string> &filename_data); 

//============================================================================
int main(int argc, char **argv) {
  int random_seed = lbann_default_random_seed;
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  int np = comm->get_procs_in_world();

  try {

    if (np!= 1) {
      LBANN_ERROR("please run with a single processor");
    }

    options *opts = options::get();
    opts->init(argc, argv);

    // check for proper invocation, print help message
    if (opts->get_bool("h") || opts->get_bool("help") || argc == 1) {
      cout << help_msg();
      return EXIT_FAILURE;
    }

    // sanity checks
    check_cmd_line();

    // ensure we have enough samples to fullfill the requirements
    sanity_test_request();

    // maps filename to { sample_ids }
    unordered_map<string, unordered_set<string>> sample_mapping;
    // maps filename to  [ sample_ids ]
    unordered_map<string, vector<string>> sample_mapping_v;
    // maps a sampleID to a local idex
    unordered_map<string, int> string_to_index;
    // note: the above mappings contain sample IDs for all samples,
    //        whether successful or failed

    read_mapping_file(sample_mapping, sample_mapping_v, string_to_index);

    unordered_map<string, unordered_set<int>> index_map_keep;
    unordered_map<string, unordered_set<int>> index_map_exclude;
    std::unordered_map<std::string, std::string> filename_data;
    build_index_maps(sample_mapping, index_map_keep, index_map_exclude, string_to_index, filename_data);

    // divide the selected samples into num_list sets
    int num_lists = opts->get_int("num_lists");
    vector<unordered_map<string, unordered_set<int>>> subsets(num_lists);
    divide_selected_samples(index_map_keep, subsets);

    const string output_dir = opts->get_string("output_dir");
    const string output_base = opts->get_string("output_base_fn");
    for (int n=0; n<num_lists; n++) {
      write_sample_list(n, subsets, sample_mapping_v, filename_data);
    }

  } catch (lbann::exception& e) {
    if (options::get()->get_bool("stack_trace_to_file")) {
      ostringstream ss("stack_trace");
      const auto& rank = get_rank_in_world();
      if (rank >= 0) {
        ss << "_rank" << rank;
      }
      ss << ".txt";
      ofstream fs(ss.str());
      e.print_report(fs);
    }
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}

// sanity check the cmd line
void check_cmd_line() {
  options *opts = options::get();
  stringstream err;
  if (! (opts->has_string("index_fn") && opts->has_string("mapping_fn")
         && opts->has_int("num_samples_per_list") && opts->has_int("num_lists")
         && opts->has_int("random_seed")
         && opts->has_string("output_dir") && opts->has_string("output_base_fn"))) {
    cout << help_msg();
    exit(0);
  }
}

string help_msg() {
      stringstream err;
      err << "usage: select_samples --index_fn=<string> --sample_mapping_fn=<string> --num_samples_per_list=<int> --num_lists --output_dir=<string> --output_base_name=<string> --random_seed=<int>\n\n";
      err << "example invocation:\n";
      err << "select_samples \n";
      err << "  --index_fn=/p/gpfs1/brainusr/datasets/10MJAG/1M_B/index.txt\n";
      err << "  --mapping_fn=/p/gpfs1/brainusr/datasets/10MJAG/1M_B/id_mapping.txt\n";
      err << "  --num_samples_per_list=1000\n";
      err << "  --num_lists=4\n";
      err << "  --output_dir=/p/gpfs1/brainusr/datasets/10MJAG/1M_B\n";
      err << "  --output_base_fn=my_samples.txt\n";
      err << "  --random_seed=42\n";
      err << "\n\n";
      return err.str();
}

void read_mapping_file(unordered_map<string, unordered_set<string>> &sample_mapping, unordered_map<string, vector<string>> &sample_mapping_v, unordered_map<string, int>& string_to_index) {
  cerr << "starting read_mapping_file\n";
  const string mapping_fn = options::get()->get_string("mapping_fn");
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
        LBANN_ERROR("duplicate sample_ID: " + to_string(sample_id) + " in file: " + filename);
      }
      string_to_index[sample_id] = hh++;
    }
  }
  in.close();
  cerr << "  FINISHED reading sample mapping: num lines processed: " << n << "\n";
}

// build two maps: <string, set<int>> maps a filename to the
// set of indices (not sample_ids; that comes later!) that are to be
// included and excluded
void build_index_maps(
  unordered_map<string, unordered_set<string>> &sample_mapping, 
  unordered_map<string, unordered_set<int>> &index_map_keep, 
  unordered_map<string, unordered_set<int>> &index_map_exclude,
  unordered_map<string, int>& string_to_index,
  unordered_map<string, string> &data) {

  cout << "starting build_index_maps\n";

  int samples_per_list = options::get()->get_int("num_samples_per_list");
  int num_lists = options::get()->get_int("num_lists");
  size_t num_samples = samples_per_list * num_lists;

  //open input file
  const string index_fn = options::get()->get_string("index_fn").c_str();
  ifstream in(index_fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open " + index_fn + " for reading");
  }

  string line;
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

  cerr << "generating random indicess ...\n";
  unordered_set<int> random_indices;
  srandom(options::get()->get_int("seed"));
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
    string sample_id;
    while (s >> sample_id) {
      if (sample_mapping[fn].find(sample_id) == sample_mapping[fn].end()) {
        LBANN_ERROR("failed to find " + to_string(sample_id) + " in sample_mapping");
      }
      index_map_exclude[fn].insert(string_to_index[sample_id]);
    }
    if (index_map_exclude[fn].size() != bad) {
      LBANN_ERROR("exclude.size(): " + to_string(index_map_exclude[fn].size()) + " should be: " + to_string(bad) + " but isn't\n");
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
  cout << "  FINISHEDbuild_index_maps\n";
}

void sanity_test_request() {
  const string index_fn = options::get()->get_string("index_fn").c_str();
  ifstream in(index_fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open " + index_fn + " for reading");
  }

  string line;
  getline(in, line);
  if (line != "CONDUIT_HDF5_EXCLUSION") {
    LBANN_ERROR("error: 1st line in index file must contain: CONDUIT_HDF5_EXCLUSION\n");
  }

  int num_valid, num_invalid, num_files;
  in >> num_valid >> num_invalid >> num_files;
  int samples_per_list = options::get()->get_int("num_samples_per_list");
  int num_lists = options::get()->get_int("num_lists");
  int num_samples = samples_per_list * num_lists;
  if (num_samples > num_valid) {
    LBANN_ERROR("you requested a total of " + to_string(num_samples) + " samples, but only " + to_string("num_valid") + " are available");
  }
}

void divide_selected_samples(
    const unordered_map<string, unordered_set<int>> &index_map_keep,
    vector<unordered_map<string, unordered_set<int>>> &sets) {
  size_t samples_per_list = options::get()->get_int("num_samples_per_list");
  size_t which = 0;
  size_t count = 0;
  size_t total = 0;
  for (auto &it : index_map_keep) {
    const string &filename = it.first;
    const unordered_set<int> &sample_ids = it.second;
    for (auto &it2 : sample_ids) {
      sets[which][filename].insert(it2);
      ++total;
      ++count;
      if (count == samples_per_list) {
        count = 0;
        ++which;
      }
    }
  }

  if (which != sets.size()) {
    LBANN_ERROR("which != sets.size()");
  }
  if (total != samples_per_list * sets.size()) {
    LBANN_ERROR("samples_per_list * sets.size()");
  }
}

void write_sample_list(
    int n, 
    vector<unordered_map<string, unordered_set<int>>> &subsets, 
    unordered_map<string, vector<string>> &sample_mapping_v,
    std::unordered_map<std::string, std::string> &filename_data) {
  const string dir = options::get()->get_string("output_dir");
  const string fn = options::get()->get_string("output_base_fn");
  stringstream s;
  s << dir << '/' << "t_" << n << '_' << fn;
  ofstream out(s.str().c_str());
  if (!out) {
    LBANN_ERROR("failed to open " + s.str() + " for writing");
  }
  cout << "WRITING output file: " << s.str() << endl;

  out << "CONDUIT_HDF5_INCLUSION\n";
  stringstream s2;
  size_t total_good = 0;
  size_t total_bad = 0;
  size_t num_include_files = 0;
  stringstream sout;
  for (auto &t : subsets[n]) {
    const string &filename = t.first;
    if (filename_data.find(filename) == filename_data.end()) {
      #if 0
      err << "filename_data.find(" << filename << ") failed\n";
      for (auto tt : filename_data) {
        err << tt.first << "\n";
      }
      LBANN_ERROR(err.str());
      #endif
      LBANN_ERROR("filename_data.find(" + filename + ") failed");
    }

    // get total samples for the current file
    stringstream s5(filename_data[filename]);
    int good, bad;
    string fn_discard;
    s5 >> fn_discard >> good >> bad;
    size_t total = good+bad;
    const unordered_set<int> &include_me = t.second;
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

  const string index_fn = options::get()->get_string("index_fn").c_str();
  ifstream in(index_fn.c_str());
  string line;
  getline(in, line);
  int num_valid, num_invalid, num_files;
  in >> num_valid >> num_invalid >> num_files;
  getline(in, line);  //discard newline
  string base_dir;
  getline(in, base_dir);

  out << total_good << " " << total_bad << " " << num_include_files
      << "\n" << base_dir << "\n" << sout.str();
}
