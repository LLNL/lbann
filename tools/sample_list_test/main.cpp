#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "sample_list_jag.hpp"
#include <unordered_set>

using namespace lbann;

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "usage : > " << argv[0] << " sample_list_file num_ranks" << std::endl;
    return 0;
  }

  // The file name of the sample file list
  std::string sample_list_file(argv[1]);

  // The number of ranks to divide samples with
  int num_ranks = atoi(argv[2]);

  sample_list_jag sn;
  sn.set_num_partitions(num_ranks);
/*
  sn.load(sample_list_file);
*/
  std::ifstream istr(sample_list_file);
  sn.get_samples_per_file(istr);
  // Create a copy of the sample file list read for verification
  //sn.write("copy_of_" + sample_list_file);


  for(int i = 0; i < num_ranks; ++i) {
    std::string sstr;
    sn.to_string(static_cast<size_t>(i), sstr);
    std::cout << sstr;
  }
  return 0;
}


