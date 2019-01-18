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
#include "lbann/utils/jag_utils.hpp"
#include "lbann/utils/options.hpp"


using namespace std;
using namespace lbann;

int main(int argc, char **argv) {
  int random_seed = lbann_default_random_seed;
  lbann_comm *comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  options *opts = options::get();
  opts->init(argc, argv);

  // sanity check the cmd line
  if (argc != 2) {
    if (master) {
      cerr << "\nusage: " << argv[0] << " --base_dir=<string>\n"
           << "assumes: the file '<base_dir>/index.txt' exists\n"
           << "output: writes the file <base_dir>/id_mapping.txt\n\n";
    }
    finalize(comm);
    return(0);
  }

  // get list of conduit filenames
  if (master) cerr << "reading filelist\n";
  vector<string> filenames;
  string base_dir = opts->get_string("base_dir");
  if (base_dir.back() != '/') {
    base_dir += '/';
  }
  char b[1024];
  sprintf(b, "%s/index.txt", base_dir.c_str());
  std::string fn;
  std::ifstream in(b);
  if (!in) LBANN_ERROR("can't open file for writing");
  std::string line;
  getline(in, line);
  getline(in, line);
  getline(in, line);
  while (getline(in, line)) {
    if (line.size() < 4) break;
    std::stringstream s(line);
    s >> fn;
    filenames.push_back(fn);
  }
  in.close();

  // each proc opens a file for output
  if (master) cerr << "opening output files\n";
  sprintf(b, "%s/TMP_%d", base_dir.c_str(), rank);
  ofstream out(b);
  if (!out) LBANN_ERROR("can't open file for writing");

  // each proc builds a map: sample_id -> local index, for the
  // conduit files for which it's responsible
  size_t q = 0;
  if (master) cerr << "building map\n";
  for (size_t j=rank; j<filenames.size(); j+= np) {
    out << filenames[j] << " ";
    ++q;
    if (q % 10 == 0) cout << rank << " :: " << q/10 << " *10 processed\n";
    const std::string f_name(base_dir + filenames[j]); 
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( f_name );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    for (size_t h=0; h<cnames.size(); h++) {
      out << cnames[h] << " ";
    }
    out << "\n";
  }

  out.close();
  MPI_Barrier(MPI_COMM_WORLD);

  if (master) {
    stringstream s3;
    s3 << "cat " << base_dir << "/TMP_* > " << base_dir << "/id_mapping.txt";
    int r = system(s3.str().c_str());
    if (r) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s3.str());
    }

    s3.clear();
    s3.str("");
    s3 << "chmod 660 " << base_dir << "/id_mapping.txt; chgrp brain " << base_dir << "/id_mapping.txt";
    r = system(s3.str().c_str());
    if (r) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s3.str());
    }

    s3.clear();
    s3.str("");

    s3 << "rm -f " << base_dir << "/TMP_*";
    r = system(s3.str().c_str());
    if (r) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s3.str());
    }
  }

  finalize(comm);
}
