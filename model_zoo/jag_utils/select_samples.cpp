#include <algorithm>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <unordered_set>
#include <set>

using namespace std;

int main(int argc, char **argv) {
  if (argc != 5) {
    cerr << "\nusage: " << argv[0] << " master_index_fn num_samples output_fn random_seed\n\n";
    exit(9);
  }

  ifstream in(argv[1]);
  if (!in) {
    cerr << "failed to open " << argv[1] << " for reading\n";
    exit(9);
  }
  ofstream out(argv[3]);
  if (!out) {
    cerr << "failed to open " << argv[3] << " for writing\n";
    exit(9);
  }
  char b[1024];
  sprintf(b, "%s_bar", argv[3]);
  ofstream out_bar(b);
  if (!out) {
    cerr << "failed to open " << argv[3] << "_bar for writing\n";
    exit(9);
  }


  string line;
  getline(in, line);
  if (line != "CONDUIT_HDF5_EXCLUSION") {
    cerr << "error: 1st line in index file must contain: CONDUIT_HDF5_EXCLUSION\n";
    exit(9);
  }
  string base_dir;
  int num_valid, num_invalid, num_files;
  in >> num_valid >> num_invalid >> num_files;
  getline(in, line);
  getline(in, base_dir);
  cout << "input index file contains " << num_valid << " valid and " << num_invalid << " samples\n";

  // generate random indices; note that these are global indices
  cout << "generating random indicess ...\n";
  int n_samples = atoi(argv[2]);
  int seed = atoi(argv[4]);
  unordered_set<int> random_indices;
  srandom(seed);
  while (true) {
    int v = random() % num_valid;
    random_indices.insert(v);
    if (random_indices.size() == n_samples) {
      break;
    }
  }
  cout << "DONE!\n";

  // loop over each entry from in input index file;
  // determine which, if any, local indices will be
  // added to the INCLUSION index
  int first = 0;
  int i = 0;
  int good, bad, total;
  int x;
  string fn;

  int valid_i = 0;
  int invalid_i = 0;
  int files_i = 0;

  int valid_e = 0;
  int invalid_e = 0;
  int files_e = 0;

  stringstream e_stream;
  stringstream i_stream;

  vector<int> keepme;
  vector<int> unused;
  unordered_set<int> exclude;

  while (! in.eof()) {
    getline(in, line);
    if (!line.size()) {
      break;
    }
    ++i;
    if (i % 1000 == 0) cout << i/1000 << "K input lines processed\n";
    stringstream s(line);
    s >> fn >> good >> bad;
    total = good+bad;
    int local_valid_index = 0;
    exclude.clear();
    while (s >> x) {
      exclude.insert(x);
    }
    if (exclude.size() != bad) {
      cerr << "exclude.size(): " << exclude.size() << " should be: " << bad << " but isn't\n";
      exit(9);
    }

    keepme.clear();
    unused.clear();
    for (int local_index=0; local_index<total; local_index++) {
      if (exclude.find(local_index) == exclude.end()) {
        int global_idx = local_valid_index+first;
        if (random_indices.find(global_idx) != random_indices.end()) {
          keepme.push_back(local_index);
        } else {
          unused.push_back(local_index);
        }
        ++local_valid_index;
      }
    }
    first += good;

    valid_e += unused.size();
    invalid_e += bad+keepme.size();
    ++files_e;

    if (keepme.size()) {
    valid_i += keepme.size();
    invalid_i += (bad+unused.size());
    ++files_i;
      i_stream << fn << " " << keepme.size() << " " << bad+unused.size();
      for (auto t : keepme) {
        i_stream << " " << t;
      }
      i_stream << endl;
    }

    e_stream << fn << " " << unused.size() << " " << bad+keepme.size();
    vector<int> v2;
    v2.reserve(bad+sizeof(keepme));
    for (auto t : keepme) {
      v2.push_back(t);
    }
    for (auto t : exclude) {
      v2.push_back(t);
    }
    sort(v2.begin(), v2.end());
    for (auto t : v2) {
      e_stream << " " << t;
    }
    e_stream << "\n";
  }

  out << "CONDUIT_HDF5_INCLUSION\n";
  out << valid_i << " " << invalid_i << " " << files_i << "\n";
  out << base_dir << "\n" << i_stream.str();
  out.close();

  out_bar << "CONDUIT_HDF5_EXCLUSION\n";
  out_bar << valid_e << " " << invalid_e << " " << files_e << "\n";
  out_bar << base_dir << "\n" << e_stream.str();
  out_bar.close();
}
