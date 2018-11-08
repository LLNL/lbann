#include "sample_list.hpp"
#include <deque>
#include <memory>
#include <mpi.h>
#include <cstdio>

using namespace lbann;

MPI_Status status;

struct send_request {
  int m_receiver;
  MPI_Request m_mpi_request;
  std::shared_ptr<std::string> m_data;
  unsigned long m_buf_size;

  send_request() {
    m_data = std::make_shared<std::string>();
  }

  void set_receiver(int recv) {
    m_receiver = recv;
  }

  int get_receiver() const {
    return m_receiver;
  }

  MPI_Request& request() {
    return m_mpi_request;
  }

  std::string* data() const {
    return m_data.get();
  }

  unsigned long& size() {
    m_buf_size = static_cast<unsigned long>(m_data->size());
    return m_buf_size;
  }
};


void handle_mpi_error(int ierr) {
  int errclass, resultlen;;
  char err_buffer[MPI_MAX_ERROR_STRING];
 
  if (ierr != MPI_SUCCESS) {
    MPI_Error_class(ierr, &errclass);
    if (errclass == MPI_ERR_RANK) {
      fprintf(stderr, "Invalid rank used in MPI send call\n");
      MPI_Error_string(ierr, err_buffer, &resultlen);
      fprintf(stderr,err_buffer);
      MPI_Finalize();             /* abort*/
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cout << "usage : > " << argv[0] << " sample_list_file num_ranks" << std::endl;
    return 0;
  }

  std::string my_samples;

  // The number of ranks to divide samples with
  int num_ranks = 1;
  int my_rank = 0;
  int ierr;
  int root_rank = 0;
  int size_tag = 0;
  int data_tag = 1;
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &num_ranks);
  MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
  
  if (my_rank == root_rank) {
    // The file name of the sample file list
    std::string sample_list_file(argv[1]);

    sample_list<> sn;
    sn.set_num_partitions(num_ranks);
    sn.load(sample_list_file);
    std::deque< send_request > send_requests;

    // Start of serialization and transmission
    MPI_Barrier(comm);

    for(int i = 0; i < num_ranks; ++i) {
      if (i == root_rank) {
        sn.to_string(static_cast<size_t>(root_rank), my_samples);
        continue;
      }

      send_requests.emplace_back();
      auto& req0 = send_requests.back();
      send_requests.emplace_back();
      auto& req1 = send_requests.back();
      req0.set_receiver(i);
      req1.set_receiver(i);
      std::string& sstr = *(req1.data());

      sn.to_string(static_cast<size_t>(i), sstr);
      unsigned long& bufsize = req1.size();

      ierr = MPI_Isend(reinterpret_cast<void*>(&bufsize), 1,
                       MPI_UNSIGNED_LONG, i, size_tag, comm, &(req0.request())); 
      handle_mpi_error(ierr);

      ierr = MPI_Isend(reinterpret_cast<void*>(const_cast<char*>(sstr.data())), static_cast<int>(sstr.size()),
                       MPI_BYTE, i, data_tag, comm, &(req1.request())); 
      handle_mpi_error(ierr);

      const int n_prev_reqs = static_cast<int>(send_requests.size() - 2);

      for (int j = 0; j < n_prev_reqs; ++j) {
        MPI_Status status;
        int flag;
        auto& req = send_requests.front();

        MPI_Test(&(req.request()), &flag, &status);

        if (!flag) {
          break;
        }
        std::cout << "completnig Isend for recv by rank " << req.get_receiver() << std::endl;
        send_requests.pop_front();
      }
    }

    for (auto& req: send_requests) {
      MPI_Status status;
      MPI_Wait(&(req.request()), &status);
      std::cout << "finally completnig Isend for recv by rank " << req.get_receiver() << std::endl;
    }

    send_requests.clear();
  } else {
    // Start of serialization and transmission
    MPI_Barrier(comm);
    MPI_Status status;
    unsigned long bufsize = 0u;
    ierr = MPI_Recv(reinterpret_cast<void*>(&bufsize), 1,
                    MPI_UNSIGNED_LONG, root_rank, size_tag, comm, &status);
    handle_mpi_error(ierr);

    my_samples.resize(bufsize);

    ierr = MPI_Recv(reinterpret_cast<void*>(&my_samples[0]), static_cast<int>(bufsize),
                    MPI_BYTE, root_rank, data_tag, comm, &status);
    handle_mpi_error(ierr);
  }

  // End of serialization and transmission
  MPI_Barrier(MPI_COMM_WORLD);

  for(int i = 0; i < num_ranks; ++i) {
    if (i == my_rank) {
      std::cout << my_samples;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Finalize();

  return 0;
}


