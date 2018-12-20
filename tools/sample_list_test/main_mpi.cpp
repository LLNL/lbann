#include "lbann/utils/timer.hpp"
#include "sample_list_jag.hpp"
#include <deque>
#include <memory>
#include <mpi.h>
#include <cstdio>

using namespace lbann;


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


template <typename SN>
void distribute_sample_list(const sample_list<SN>& sn,
                            std::string& my_samples,
                            MPI_Comm& comm);


void write_sample_list(const std::string& my_samples, MPI_Comm& comm);
void print_sample_list(const std::string& my_samples, MPI_Comm& comm);
void deserialize_sample_list(const std::string& my_samples, MPI_Comm& comm);


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
  MPI_Comm_size(comm, &num_ranks);

  double tm1 = get_time();
  sample_list<std::string> sn;
  sn.set_num_partitions(num_ranks);
  sn.load(sample_list_file);
  distribute_sample_list(sn, my_samples, comm);

  write_sample_list(my_samples, comm);
  double tm2 = get_time();

  deserialize_sample_list(my_samples, comm);
  double tm3 = get_time();
  std::cout << "Time: " << tm2 - tm1 << " " << tm3 - tm2 << std::endl;

  MPI_Finalize();

  return 0;
}


template <typename SN>
void distribute_sample_list(const sample_list<SN>& sn,
                            std::string& my_samples,
                            MPI_Comm& comm) {
  int num_ranks = 1;
  int my_rank = 0;
  int root_rank = 0;
  int size_tag = 0;
  int data_tag = 1;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &num_ranks);
  MPI_Errhandler_set(comm, MPI_ERRORS_RETURN);

  if (my_rank == root_rank) {

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

      int ierr;
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
        send_requests.pop_front();
      }
    }

    for (auto& req: send_requests) {
      MPI_Status status;
      MPI_Wait(&(req.request()), &status);
    }

    send_requests.clear();
  } else {
    // Start of serialization and transmission
    MPI_Barrier(comm);

    MPI_Status status;
    int ierr;
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
}


void write_sample_list(const std::string& my_samples, MPI_Comm& comm) {
  int num_ranks = 1;
  int my_rank = 0;
  MPI_Comm_size(comm, &num_ranks);
  MPI_Comm_rank(comm, &my_rank);
  std::string out_filename = "ample_list." + std::to_string(my_rank) + ".txt";
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
      std::cout << my_samples;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}


void deserialize_sample_list(const std::string& my_samples, MPI_Comm& comm) {
  sample_list<> sn;
  sn.load_from_string(my_samples);
  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);
}
