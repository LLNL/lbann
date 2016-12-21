#include "lbann/lbann.hpp"
#include "lbann/utils/lbann_quantizer.hpp"
#include "lbann/utils/lbann_timer.hpp"

using namespace lbann;

const int num_trials = 20;

std::vector<double> test_mpi_allreduce(lbann_comm* comm, DistMat& mat) {
  std::vector<double> times;
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    comm->intermodel_sum_matrix(mat);
    double tot = get_time() - start;
    times.push_back(tot);
    comm->global_barrier();
  }
  return times;
}

void dt_sum(void* in_, void* inout_, int* len, MPI_Datatype* dtyp) {
  DataType* in = (DataType*) in_;
  DataType* inout = (DataType*) inout_;
  for (int i = 0; i < *len; ++i) {
    inout[i] += in[i];
  }
}

std::vector<double> test_mpi_datatype_allreduce(lbann_comm* comm, DistMat& mat) {
  std::vector<double> times;
  MPI_Datatype cust_dt;
  MPI_Type_contiguous(1, MPI_FLOAT, &cust_dt);
  MPI_Type_commit(&cust_dt);
  MPI_Op cust_op;
  MPI_Op_create(dt_sum, 1, &cust_op);
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    MPI_Allreduce(MPI_IN_PLACE, mat.Buffer(),
                  mat.LocalHeight() * mat.LocalWidth(),
                  cust_dt, cust_op, MPI_COMM_WORLD);
    double tot = get_time() - start;
    times.push_back(tot);
    comm->global_barrier();
  }
  return times;
}

std::vector<double> test_cust_allreduce(lbann_comm* comm, DistMat& mat) {
  lbann_quantizer quantizer;
  std::vector<double> times;
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    quantizer.intermodel_sum(comm, mat);
    double tot = get_time() - start;
    times.push_back(tot);
    comm->global_barrier();
  }
  return times;
}

void print_stats(const std::vector<double>& times) {
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / times.size();
  auto minmax = std::minmax_element(times.begin(), times.end());
  double sqsum = 0.0;
  for (const auto& t : times) {
    sqsum += (t - mean) * (t - mean);
  }
  double stdev = std::sqrt(sqsum / (times.size() - 1));
  std::cout << "\tMean: " << mean << std::endl;
  std::cout << "\tMin: " << *(minmax.first) << std::endl;
  std::cout << "\tMax: " << *(minmax.second) << std::endl;
  std::cout << "\tStdev: " << stdev << std::endl;
  std::cout << "\tRaw: ";
  for (const auto& t : times) {
    std::cout << t << ", ";
  }
  std::cout << std::endl;
}

void test_mat(lbann_comm* comm, DistMat& mat) {
  auto mpi_times = test_mpi_allreduce(comm, mat);
  if (comm->am_world_master()) {
    std::cout << "MPI (" << mat.Height() << "x" << mat.Width() << "):" <<
      std::endl;
    print_stats(mpi_times);
  }
  auto mpi_dt_times = test_mpi_datatype_allreduce(comm, mat);
  if (comm->am_world_master()) {
    std::cout << "MPI DataType (" << mat.Height() << "x" << mat.Width() <<
       "):" << std::endl;
    print_stats(mpi_dt_times);
  }
  auto cust_times = test_cust_allreduce(comm, mat);
  if (comm->am_world_master()) {
    std::cout << "Custom (" << mat.Height() << "x" << mat.Width() << "):" <<
      std::endl;
    print_stats(cust_times);
  }
}

int main(int argc, char** argv) {
  El::Initialize(argc, argv);
  // 1 because we use MPI_COMM_WORLD above.
  lbann_comm* comm = new lbann_comm(1);
  for (int mat_size = 64; mat_size <= 16384; mat_size *= 2) {
    DistMat mat(comm->get_model_grid());
    El::Uniform(mat, mat_size, mat_size, 0.0f, 4.0f);
    test_mat(comm, mat);
  }
  El::Finalize();
}
