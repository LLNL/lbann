#include <algorithm>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include "lbann/utils/options.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/comm.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using lbann::options;

//============================================================================
// sanity checks the cmd line
void check_cmd_line();

// returns the help message
string help_msg();

// tests that the output dir exists and is writable,
// and creates it if otherwise
void test_output_dir();

// tests that there are sufficient samples to build the lists
// (i.e, num_lists*num_samples_per_list must not be greater than
// the total number of (successful) samples
void sanity_test_request();

// constructs various mappings from the mapping file
void read_mapping_file(
  unordered_map<string, unordered_set<string>> &sample_mapping, 
  unordered_map<string, vector<string>> &sample_mapping_v, 
  unordered_map<string, int>& string_to_index);

// constructs various mappings from the index file
void build_index_maps(
  unordered_map<string, unordered_set<int>> &index_map_keep, 
  unordered_map<string, unordered_set<int>> &index_map_exclude,
  unordered_map<string, int> &string_to_index,
  unordered_map<string, string> &filename_data);

// partition the sample IDs in index_map_keep into n sets;
// on entry, sets.size() = num_lists
void divide_selected_samples(
  const unordered_map<string, unordered_set<int>> &index_map_keep,
  vector<unordered_map<string, unordered_set<int>>> &sets); 

// write the n-th sample list to file
void write_sample_list(
    int n, 
    const vector<unordered_map<string, unordered_set<int>>> &subsets, 
    const unordered_map<string, vector<string>> &sample_mapping_v,
    const std::unordered_map<std::string, std::string> &filename_data); 

void write_exclusion_file(
  const unordered_map<string, unordered_set<int>> index_map_exclude,
  const unordered_map<string, vector<string>> &sample_mapping_v, 
  const unordered_map<string, string> &filename_data);
//============================================================================
int main(int argc, char **argv) {
  int random_seed = lbann::lbann_default_random_seed;
  lbann::world_comm_ptr comm = lbann::initialize(argc, argv, random_seed);
  int np = comm->get_procs_in_world();

  try {

    if (np != 1) {
      LBANN_ERROR("please run with a single processor");
    }

    options *opts = options::get();
    opts->init(argc, argv);

    // check for proper invocation, print help message
    if (opts->get_bool("h") || opts->get_bool("help") || argc == 1) {
      cout << help_msg();
      return EXIT_FAILURE;
    }

    // check for proper invocation
    check_cmd_line();

    // check that output directory exists and is writable,
    // and creates it if otherwise
    test_output_dir();

    // ensure we have enough samples to fullfill the requirements
    sanity_test_request();

    // maps a sample_id filename to the set of sample IDs
    unordered_map<string, unordered_set<string>> sample_mapping;
    // maps  a sample_id filename to a list of sample IDs
    unordered_map<string, vector<string>> sample_mapping_v;
    // maps a sampleID to a local idex
    unordered_map<string, int> string_to_index;

    read_mapping_file(sample_mapping, sample_mapping_v, string_to_index);

    // maps a samole_id filename to a set of randomly selected sample_ids 
    unordered_map<string, unordered_set<int>> index_map_keep;
    // maps a samole_id filename to the set of sample_ids that have not been randomly selscted
    unordered_map<string, unordered_set<int>> index_map_exclude;
    std::unordered_map<std::string, std::string> filename_data;
    build_index_maps(index_map_keep, index_map_exclude, string_to_index, filename_data);

    // partition the randomly selected samples into "num_lists" sets
    int num_lists = opts->get_int("num_lists");
    vector<unordered_map<string, unordered_set<int>>> subsets(num_lists);
    divide_selected_samples(index_map_keep, subsets);

    write_exclusion_file(index_map_exclude, sample_mapping_v, filename_data);

    // write the sample lists
    for (int n=0; n<num_lists; n++) {
      write_sample_list(n, subsets, sample_mapping_v, filename_data);
    }

    cout << "SUCESS - FINISHED!\n";

  } catch (lbann::exception& e) {
    if (options::get()->get_bool("stack_trace_to_file")) {
      ostringstream ss("stack_trace");
      const auto& rank = lbann::get_rank_in_world();
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
    if (!opts->has_string("index_fn")) {
      cout << "missing --index_fn=<string> \n";
    }
    if (!opts->has_string("mapping_fn")) {
      cout << "missing --mapping_fn=<string> \n";
    }
    if (!opts->has_string("num_samples_per_list")) {
      cout << "missing --num_samples_per_list=<int> \n";
    }
    if (!opts->has_string("num_lists")) {
      cout << "missing --num_lists=<int> \n";
    }
    if (!opts->has_string("random_seed")) {
      cout << "missing --random_seed=<int> \n";
    }
    if (!opts->has_string("output_dir")) {
      cout << "missing --output_dir=<string> \n";
    }
    if (!opts->has_string("output_base_fn")) {
      cout << "missing --output_base_fn=<string> \n";
    }
    cout << "\n";
    exit(0);
  }
}

string help_msg() {
      stringstream err;
      err << "usage: select_samples --index_fn=<string> --sample_mapping_fn=<string> --num_samples_per_list=<int> --num_lists --output_dir=<string> --output_base_name=<string> --random_seed=<int>\n\n";
      err << "example invocation:\n";
      err << "select_samples \\\n";
      err << "  --index_fn=/p/gpfs1/brainusr/datasets/100M/index.txt \\\n";
      err << "  --mapping_fn=/p/gpfs1/brainusr/datasets/100M/id_mapping.txt \\\n";
      err << "  --num_samples_per_list=100000 \\\n";
      err << "  --num_lists=640 \\\n";
      err << "  --output_dir=/p/gpfs1/brainusr/datasets/100M/1M_B \\\n";
      err << "  --output_base_fn=my_samples.txt \\\n";
      err << "  --random_seed=42\n";
      err << "\n\n";
      err << "NOTE: output directory must exist prior to running this code\n";

      return err.str();
}

void read_mapping_file(unordered_map<string, unordered_set<string>> &sample_mapping, unordered_map<string, vector<string>> &sample_mapping_v, unordered_map<string, int>& string_to_index) {
  cout << "starting read_mapping_file\n";
  double tm1 = lbann::get_time();
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
        LBANN_ERROR("duplicate sample_ID: ", sample_id, " in file: ", filename);
      }
      string_to_index[sample_id] = hh++;
    }
  }
  in.close();
  double tm2 = lbann::get_time() - tm1;
  cout << "  FINISHED reading sample mapping: num lines processed: " << n << "; time: " << tm2 << "\n";
}

// build two maps: <string, set<int>> maps a filename to the
// set of indices (not sample_ids; that comes later!) that are to be
// included and excluded
void build_index_maps(
  unordered_map<string, unordered_set<int>> &index_map_keep, 
  unordered_map<string, unordered_set<int>> &index_map_exclude,
  unordered_map<string, int>& string_to_index,
  unordered_map<string, string> &filename_data) {

  cout << "starting build_index_maps\n";
  double tm1 = lbann::get_time();

  int samples_per_list = options::get()->get_int("num_samples_per_list");
  int num_lists = options::get()->get_int("num_lists");
  size_t num_samples = samples_per_list * num_lists;

  //open input file
  const string index_fn = options::get()->get_string("index_fn").c_str();
  ifstream in(index_fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", index_fn, " for reading");
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
  options::get()->set_option("base_dir", base_dir);
  cout << "input index file contains " << num_valid << " valid samples\n";

  cout << "generating random indices ...\n";
  double tm2 = lbann::get_time();
  unordered_set<int> random_indices;
  srandom(options::get()->get_int("random_seed"));
  while (true) {
    int v = random() % num_valid;
    random_indices.insert(v);
    if (random_indices.size() == num_samples) {
      break;
    }
  }
  cout << "  FINISHED generating random indices; time: " << lbann::get_time() - tm2 << endl;
  cout << "selecting samples based on random indices\n";
  double tm3 = lbann::get_time();

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
    if (num_files % 1000 == 0) cout << num_files/1000 << "K input lines processed\n";
    stringstream s(line);
    s >> fn >> good >> bad;
    filename_data[fn] = line;
    const int total = good+bad;
    index_map_exclude[fn];
    index_map_keep[fn];
    string sample_id;

    while (s >> sample_id) {
      index_map_exclude[fn].insert(string_to_index[sample_id]);
    }
    if (index_map_exclude[fn].size() != bad) {
      LBANN_ERROR("exclude.size(): ", index_map_exclude[fn].size(), " should be: ", bad, " but isn't\n");
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
  cout << "FINISHED selecting samples based on random indices; time: " << lbann::get_time() - tm3 << endl;

  if (index_map_exclude.size() != index_map_keep.size()) {
    LBANN_ERROR("index_map_exclude.size() != index_map_keep.size()");
  }
  cout << "  FINISHED build_index_maps; time: " << lbann::get_time() - tm1 << endl;
}

void sanity_test_request() {
  const string index_fn = options::get()->get_string("index_fn").c_str();
  ifstream in(index_fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", index_fn, " for reading");
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
    LBANN_ERROR("you requested a total of ", num_samples, " samples, but only ", num_valid, " are available");
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
    const vector<unordered_map<string, unordered_set<int>>> &subsets, 
    const unordered_map<string, vector<string>> &sample_mapping_v,
    const std::unordered_map<std::string, std::string> &filename_data) {
  const string dir = options::get()->get_string("output_dir");
  const string fn = options::get()->get_string("output_base_fn");
  stringstream s;
  s << dir << '/' << "t" << n << '_' << fn;
  ofstream out(s.str().c_str());
  if (!out) {
    LBANN_ERROR("failed to open ", s.str(), " for writing");
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

    // get total samples for the current file
    std::unordered_map<std::string, std::string>::const_iterator t4 = filename_data.find(filename);
    if (t4 == filename_data.end()) {
      LBANN_ERROR("t4 == filename_data.end()");
    }
    stringstream s5(t4->second);
    int good, bad;
    string fn2;
    s5 >> fn2 >> good >> bad;
    size_t total = good+bad;

    const unordered_set<int> &include_me = t.second;
    int included = include_me.size();
    int excluded = total - included;

    if (included) {
      ++num_include_files;
      total_good += included;
      total_bad += excluded;
      sout << filename << " " << included << " " << excluded;
      for (auto &t3 : include_me) {
        if (sample_mapping_v.find(fn2) == sample_mapping_v.end()) {
          LBANN_ERROR("failed to find the key: ", fn2, " in sample_mapping_v map");
        }  
        unordered_map<string, vector<string>>::const_iterator t5 = sample_mapping_v.find(fn2);
        if (t5 == sample_mapping_v.end()) {
          LBANN_ERROR("t5 == sample_mapping_v.end()");
        }
        if (static_cast<size_t>(t3) >= t5->second.size()) {
          LBANN_ERROR("t3 >= t5->second.size()");
        }
        sout << " " << t5->second[t3];
      }
      sout << "\n";
    }
  }

  const string base_dir = options::get()->get_string("base_dir");

  out << total_good << " " << total_bad << " " << num_include_files
      << "\n" << base_dir << "\n" << sout.str();
  out.close();
}

bool file_exists(const char *path) {
  struct stat s;
  int err = stat(path, &s);
  if (err == -1) {
    return false;
  }
  return true;
}

void make_dir(char *cpath) {
  cout << "   path doesn't exist: " << strerror(errno) << endl;
  cout << "   attempting to create path\n";
  int err = mkdir(cpath, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IXUSR | S_IXGRP);
  if (err) {
    free(cpath);
    LBANN_ERROR("mkdir failed for \"", cpath, "\"; please create this directory yourself, then rerun this program");
    cout << "   mkdir failed: " << strerror(errno) << endl;
  } else {
    cout << "   SUCCESS!\n";
    cout << "   attempting to change permissions\n";
    err = chmod(cpath, S_ISGID | S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IXUSR | S_IXGRP);
    if (err) {
      cout << "   mkdir failed: " << strerror(errno) << endl;
    } else {
      cout << "   SUCCESS!\n";
    }
  }
}

void test_output_dir() {
  cout << "\nChecking if output diretory path exists;\n"
          " if not, we'll attempt to create it.\n";
  const string dir = options::get()->get_string("output_dir");
  char *cpath = strdup(dir.c_str());
  char *pp = cpath;
  if (pp[0] == '/') {
    ++pp;
  }
  char *sp;
  int status = 0;
  while (status == 0 && (sp = strchr(pp, '/')) != 0) {
    if (sp != pp) {
      *sp = '\0';
      cout << cpath << endl;
      if (file_exists(cpath)) {
        cout << "  path exists\n";
      } else {
        make_dir(cpath);
      }
      *sp = '/';
    }
    pp = sp+1;
  }
  if (status == 0) {
    cout << cpath << endl;
    if (file_exists(cpath)) {
      cout << "  path exists\n";
    } else {
      make_dir(cpath);
    }  
  }
  free(cpath);
  cout << endl;
}


void write_exclusion_file(
  const unordered_map<string, unordered_set<int>> index_map_exclude,
  const unordered_map<string, vector<string>> &sample_mapping_v, 
  const unordered_map<string, string> &filename_data
) {
  const string dir = options::get()->get_string("output_dir");
  const string base_fn = options::get()->get_string("output_base_fn");
  stringstream s;
  s << dir << '/' << "t_" << '_' << base_fn << "_bar";
  std::cerr << "\nWRITING output bar file: " << s.str() << "\n";
  std::ofstream out(s.str().c_str());
  if (!out) {
      LBANN_ERROR("failed to open ", s.str(), " for writing\n");
  }
  out<< "CONDUIT_HDF5_EXCLUSION\n";

  std::stringstream sout;
  size_t total_good = 0;
  size_t total_bad = 0;
  size_t num_include_files = 0;

  string fn;
  int good;
  int bad;
  for (auto t : index_map_exclude) {
    const string &filename = t.first;

    // get total samples for the current file
    std::unordered_map<std::string, std::string>::const_iterator t4 = filename_data.find(filename);
    if (t4 == filename_data.end()) {
      LBANN_ERROR("t4 == filename_data.end()");
    }

    std::stringstream s5(t4->second);
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
        unordered_map<string, vector<string>>::const_iterator t5 = sample_mapping_v.find(fn);
        if (t5 == sample_mapping_v.end()) {
          LBANN_ERROR("t5 == sample_mapping_v.end())");
        }
        sout << " " << t5->second[t3];
      }
      sout << "\n";
      }
    }

    const string base_dir = options::get()->get_string("base_dir");
    out << total_good << " " << total_bad << " " << num_include_files << "\n"
        << base_dir << "\n" << sout.str();
    out.close();
}
