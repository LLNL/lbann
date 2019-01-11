#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "sample_list_jag.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/glob.hpp"
#include <mpi.h>
#include <memory>
#include <deque>
#include <cstdio>

using namespace lbann;


void write_sample_list(const std::string& my_samples, MPI_Comm& comm);
void print_sample_list(const std::string& my_samples, MPI_Comm& comm);

template<typename PACKED_T>
void all_gather_packed_lists(const PACKED_T& my_samples, PACKED_T& all_samples, std::vector<int>& displ, MPI_Comm& comm);

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "usage : > " << argv[0] << " sample_list_file(s)" << std::endl;
    return 0;
  }

  std::vector<std::string> sample_list_files = glob(argv[1]);

  if (sample_list_files.size() < 1) {
    std::cerr << "failed to get data filenames" << std::endl;
  }


  // The number of ranks to divide samples with
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);

  int num_ranks = 1;
  int my_rank = 0;
  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);

  sample_list_jag sn;

  double tm1 = get_time();
  for (size_t i = my_rank; i < sample_list_files.size(); i += num_ranks) {
    std::cout << "rank " << my_rank << " reads " << sample_list_files[i] << std::endl;
    sn.load(sample_list_files[i]);
  }

  double tm2 = get_time();

  std::string my_samples;
  sn.to_string(my_samples);

  double tm3 = get_time();

  std::string all_samples;
  std::vector<int> displ;

  all_gather_packed_lists(my_samples, all_samples, displ, comm);

  double tm4 = get_time();

  for(size_t p = 1u; p < displ.size(); ++p) {
    sample_list_jag sn_tmp;
    sn_tmp.load_from_string(all_samples.substr(displ[p-1], displ[p]));
  }

  double tm5 = get_time();

  std::cout << "Time: " << tm5 - tm1 << " (" << tm2 - tm1 << " " << tm3 - tm2 << " " << tm4 - tm3 << " " << tm5 - tm4 << ")" << std::endl;

  // dump out the result
  std::string filename_rank = lbann::modify_file_name("slist_c.txt", std::to_string(my_rank));

  std::ofstream ofile(filename_rank);
  ofile << all_samples;
  ofile.close();

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

template<typename PACKED_T>
void all_gather_packed_lists(const PACKED_T& my_samples, PACKED_T& all_samples, std::vector<int>& displ, MPI_Comm& comm) { 
  int num_ranks = 1;
  int my_rank = 0;
  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);

  int my_packed_size = static_cast<int>(my_samples.size());

  std::vector<int> packed_sizes(num_ranks);

  MPI_Allgather(&my_packed_size, 1, MPI_INT,
                &packed_sizes[0], 1, MPI_INT,
                comm);

  int total_packed_size = 0;

  displ.assign(num_ranks+1, 0);

  for (size_t i = 0u; i < packed_sizes.size(); ++i) {
    const auto sz = packed_sizes[i];
    displ[i+1] = displ[i] + sz;
  }
  total_packed_size = displ.back();

  if (total_packed_size <= 0) {
    return;
  }

  all_samples.resize(static_cast<size_t>(total_packed_size));

  int num_bytes = static_cast<int>(sizeof(decltype(my_samples.back())) * my_samples.size());

  MPI_Allgatherv(const_cast<void*>(reinterpret_cast<const void*>(&my_samples[0])),
                 num_bytes,
                 MPI_BYTE,
                 reinterpret_cast<void*>(&all_samples[0]),
                 &packed_sizes[0],
                 &displ[0],
                 MPI_BYTE,
                 comm);
}
