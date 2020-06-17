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
  world_comm_ptr comm = initialize(argc, argv, random_seed);
  bool master = comm->am_world_master();
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  options *opts = options::get();
  opts->init(argc, argv);

  // sanity check the cmd line
  if (argc < 2) {
    if (master) {
      cerr << "\nusage: " << argv[0] << " --base_dir=<string> [--hydra]\n"
           << "assumes: the file '<base_dir>/index.txt' exists\n"
           << "output: writes the file <base_dir>/id_mapping.txt\n"
           << "hydra: you must include --hydra when building a mapping for\n"
           << "       hydra conduit nodes, else the output file will be\n"
           << "       meaningless, and will result in undefined behavior.";

    }
    return(0);
  }

std::unordered_set<std::string> names;
int total = 0;

  //bool hydra = opts->get_bool("hydra");

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
  if (!in) {
    std::string fn2(b);
    LBANN_ERROR("can't open file for reading: " + fn2);
  }
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
  conduit::Node n_ok;
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
if (cnames[h].find("META") == string::npos) {
  ++total;
  if (names.find(cnames[h]) != names.end()) {
    std::cout << "XX duplicate: " << cnames[h] << "\n";
  }
  names.insert(cnames[h]);
}
      const std::string key_1 = "/" + cnames[h] + "/performance/success";
      bool good = conduit::relay::io::hdf5_has_path(hdf5_file_hnd, key_1);
      if (!good) {
        std::cerr << "missing path: " << key_1 << " (this is probably OK for hydra)\n";
        continue;
      }

      try {
        conduit::relay::io::hdf5_read(hdf5_file_hnd, key_1, n_ok);
      } catch (...) {
        std::cerr << "exception hdf5_read file: " << filenames[j] << "; key: " << key_1 << "\n";
        continue;
      }
      int success = n_ok.to_int64();
      if (success == 1) {
          // the IDs that John provided look like this:
          // 274e5a16-7c3a-11e9-90fd-0894ef80059f/runno/run0001
          // however, the top-level fields, e.g, "274e5a16-7c3a-11e9-90fd-0894ef80059f,"
          // are unique, at least for the current set of hydra bricks, so for now I'm using
          // that field as the sample_id. This has the advantage that the sample_ids are at
          // the top level, as they are for JAG samples
          out << cnames[h] << " ";

        #if 0
        if (hydra) {
          const std::string key_3 = "/" + cnames[h] + "/runno";
          if (conduit::relay::io::hdf5_has_path(hdf5_file_hnd, key_3)) {
            try {
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key_3, n_ok);
            } catch (...) {
              std::cerr << "failed to read: " << key_3 << "; continuing; this is only for hydra, and may be an error\n";
              continue;
            }
            std::string s3 = n_ok.as_string();
            out << cnames[h] << "/runno/" << s3 << " ";
          }
        }

        else {
          out << cnames[h] << " ";
        }
        #endif

      }  
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

std::cout << "\n\ntotal: " << total << " uniq: " << names.size() << "\n\n";

}
