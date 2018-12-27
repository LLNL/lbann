#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "sample_list_jag.hpp"
#include "lbann/utils/timer.hpp"
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

  double tm1 = get_time();
  // (every rank) loads sample list
  sample_list_jag sn;
  sn.set_num_partitions(num_ranks);
  sn.load(sample_list_file);

  double tm2 = get_time();
  std::cout << "Time: " << tm2 - tm1 << std::endl;

  // dump out the result
  for(int i = 0; i < num_ranks; ++i) {
    std::string sstr;
    sn.to_string(static_cast<size_t>(i), sstr);
    std::cout << sstr << std::flush;
    sn.write((size_t) i, "slist.txt");
  }
  return 0;
}
