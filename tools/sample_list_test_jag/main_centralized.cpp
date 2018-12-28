#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "sample_list_jag.hpp"
#include "lbann/utils/timer.hpp"
#include <mpi.h>
#include <memory>
#include <deque>
#include <cstdio>

using namespace lbann;


void write_sample_list(const std::string& my_samples, MPI_Comm& comm);
void print_sample_list(const std::string& my_samples, MPI_Comm& comm);


int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "usage : > " << argv[0] << " sample_list_file" << std::endl;
    return 0;
  }

  // The file name of the sample file list
  std::string sample_list_file(argv[1]);

  std::string my_samples;

  // The number of ranks to divide samples with
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);

  int num_ranks = 1;
  int my_rank = 0;
  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);

  double tm1 = get_time();
  // load sample list
  sample_list_jag sn;
  sn.set_num_partitions(num_ranks);
  if (my_rank == 0) {
    sn.load(sample_list_file);
  }

  double tm2 = get_time();

  // distribute
  lbann::distribute_sample_list(sn, my_samples, comm);

  double tm3 = get_time();

  // deserialize
  sample_list_jag sn_local;
  //sn_local.set_num_partitions(1);
  sn_local.load_from_string(my_samples);
  double tm4 = get_time();
  std::cout << "Time: " << tm4 - tm1 << " (" << tm2 - tm1 << " " << tm3 - tm2 << " " << tm4 - tm3 << ")" << std::endl;

  // dump out the result
  std::string filename_rank = lbann::modify_file_name("slist_c.txt", std::to_string(my_rank));
  sn_local.write(filename_rank);

  MPI_Finalize();

  return 0;
}


void write_sample_list(const std::string& my_samples, MPI_Comm& comm) {
  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);
  std::string out_filename = "sample_list." + std::to_string(my_rank) + ".txt";
  std::ofstream ofs(out_filename);
  ofs << my_samples;
  ofs.close();
}


void print_sample_list(const std::string& my_samples, MPI_Comm& comm) {
  int num_ranks = 1;
  int my_rank = 0;
  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);

  for(int i = 0; i < num_ranks; ++i) {
    if (i == my_rank) {
      std::cout << my_samples << std::flush;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
