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
  if (argc != 3) {
    if (master) {
      cerr << "\nusage: " << argv[0] << " --filelist=<string> --output_dir=<string>\n";
    }
    finalize(comm);
    return(0);
  }
  string dir = opts->get_string("output_dir");

  // get list of conduit filenames
  vector<string> filenames;
  read_filelist(comm, opts->get_string("filelist"), filenames);

  // each proc opens a file for output
  char b[1024];
  sprintf(b, "%s/TMP_%d", dir.c_str(), rank);
  ofstream out(b);

  // each proc builds a map: sample_id -> local index, for the
  // conduit files that for which it's responsible
  for (size_t j=rank; j<filenames.size(); j+= np) {
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( filenames[j] );
    std::vector<std::string> cnames;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
    conduit::Node n_ok;
    for (size_t h=0; h<cnames.size(); h++) {
      out << cnames[h] << " " << h << "\n";
    }
  }

  out.close();
  MPI_Barrier(MPI_COMM_WORLD);

  if (master) {
    stringstream s3;
    s3 << "cat " << dir << "/TMP_* > " << dir << "/id_mapping.txt";
    int r = system(s3.str().c_str());
    if (r) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s3.str());
    }
    s3.clear();
    s3.str("");
    s3 << "rm -f " << dir << "/TMP_*";
    r = system(s3.str().c_str());
    if (r) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s3.str());
    }
  }




  finalize(comm);
}
