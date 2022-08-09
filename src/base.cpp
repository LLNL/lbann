////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
// lbann_base .cpp - Basic definitions, functions
////////////////////////////////////////////////////////////////////////////////

#include "lbann/base.hpp"

#include <omp.h>
#if defined(LBANN_TOPO_AWARE)
#include <hwloc.h>
#if defined(HWLOC_API_VERSION) && (HWLOC_API_VERSION < 0x00010b00)
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif
#endif
#ifdef LBANN_HAS_SHMEM
#include <shmem.h>
#endif // LBANN_HAS_SHMEM

#include "lbann/comm_impl.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/stack_trace.hpp"

#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB
#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include "lbann/utils/python.hpp"
#endif
#ifdef LBANN_HAS_NVSHMEM
#include "lbann/utils/nvshmem.hpp"
#endif
#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#endif

#include <cereal/types/polymorphic.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {
lbann::lbann_comm* world_comm_ = nullptr;
MPI_Errhandler err_handle;
}// namespace <anon>

namespace lbann {
// Declare the trainer finalization. It's declared here because it is
// *not* for public consumption. It is implemented in
// src/utils/lbann_library.cpp.
void finalize_trainer();

namespace utils {
// This function def is very out-of-place... It's declared in
// "serialize_matrices.hpp"...
lbann_comm& get_current_comm() noexcept { return *world_comm_; }
}// namespace utils
}// namespace lbann

auto lbann::initialize_lbann(El::mpi::Comm&& c) -> std::unique_ptr<lbann_comm>
{

  // Parse command-line arguments and environment variables
  auto& arg_parser = global_argument_parser();
  (void) arg_parser;

  // to ensure that all the necessary infrastructure in Hydrogen and
  // Aluminum has been setup.
  El::Initialize();

  // Create a new comm object with provided MPI_Comm
  auto comm = std::make_unique<lbann_comm>(0, std::move(c));
  world_comm_ = comm.get();

  // Install MPI error handler
  //MPI_Comm_create_errhandler(lbann_mpi_err_handler, &err_handle);
  //MPI_Comm_set_errhandler(MPI_COMM_WORLD, err_handle);

#if defined(LBANN_TOPO_AWARE)
  // Determine the number of NUMA nodes present.
  hwloc_topology_t topo;
  hwloc_topology_init(&topo);
  hwloc_topology_load(topo);
  int numa_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_NUMANODE);
  if (numa_depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    std::cout << comm->get_rank_in_world() <<
              ": cannot determine hwloc NUMA-node depth" << std::endl;
  }
  int num_numa_nodes = hwloc_get_nbobjs_by_depth(topo, numa_depth);
  // Warn if there are more NUMA nodes than processes per node.
  // It's probably fine if there are more processes than NUMA nodes for now.
  // We can adjust that later when we better understand the threaded perf.
  int ppn = comm->get_procs_per_node();
  if (num_numa_nodes > ppn) {
    if (comm->get_rank_in_world() == 0) {
      std::cout << comm->get_rank_in_world() <<
                ": WARNING: node has " << num_numa_nodes <<
                " NUMA nodes but you have " << ppn << " processes per node" <<
                std::endl;
    }
  }
  hwloc_topology_destroy(topo);
#endif

#ifdef LBANN_HAS_SHMEM
  // Initialize SHMEM
  if (arg_parser.get<bool>(LBANN_OPTION_INIT_SHMEM)) {
    int threading_level = SHMEM_THREAD_MULTIPLE;
    int status = shmem_init_thread(threading_level, &threading_level);
    if (status != 0 || threading_level != SHMEM_THREAD_MULTIPLE) {
      LBANN_ERROR("error initializing OpenSHMEM");
    }
  }
#endif // LBANN_HAS_SHMEM
#ifdef LBANN_HAS_NVSHMEM
  if (arg_parser.get<bool>(LBANN_OPTION_INIT_NVSHMEM)) {
    nvshmem::initialize();
  }
#endif // LBANN_HAS_NVSHMEM

#ifdef LBANN_HAS_DISTCONV
  dc::initialize(MPI_COMM_WORLD);
#endif // LBANN_HAS_DISTCONV

  return comm;
}

auto lbann::initialize_lbann(MPI_Comm c) -> std::unique_ptr<lbann_comm>
{
  return initialize_lbann(El::mpi::Comm{c});
}

auto lbann::initialize_lbann(int argc, char** argv) -> std::unique_ptr<lbann_comm>
{
  El::Initialize(argc, argv);
  return initialize_lbann(MPI_COMM_WORLD);
}

void lbann::finalize_lbann(lbann_comm* comm) {
#ifdef LBANN_HAS_NVSHMEM
  nvshmem::finalize();
#endif // LBANN_HAS_NVSHMEM
  //MPI_Errhandler_free( &err_handle );
#ifdef LBANN_HAS_DISTCONV
  dc::finalize();
#endif
#ifdef LBANN_HAS_DNN_LIB
  dnn_lib::destroy();
#endif
#ifdef LBANN_HAS_EMBEDDED_PYTHON
  python::finalize();
#endif
#ifdef LBANN_HAS_NVSHMEM
  nvshmem::finalize();
#endif // LBANN_HAS_SHMEM
#ifdef LBANN_HAS_SHMEM
  shmem_finalize();
#endif // LBANN_HAS_SHMEM
  if (comm != nullptr) {
    delete comm;
  }
  El::Finalize();
}

auto lbann::initialize(int& argc, char**& argv) -> world_comm_ptr
{
  // Parse command-line arguments and environment variables
  auto& arg_parser = global_argument_parser();
  (void) arg_parser;

  // Initialize Elemental.
  El::Initialize(argc, argv);

  // Create a new comm object.
  // Initial creation with every process in one model.
  auto comm = world_comm_ptr{new lbann_comm(0), &lbann::finalize };
  world_comm_ = comm.get();

  // Install MPI error handler
  MPI_Comm_create_errhandler(lbann_mpi_err_handler, &err_handle);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, err_handle);

#if defined(LBANN_TOPO_AWARE)
  // Determine the number of NUMA nodes present.
  hwloc_topology_t topo;
  hwloc_topology_init(&topo);
  hwloc_topology_load(topo);
  int numa_depth = hwloc_get_type_depth(topo, HWLOC_OBJ_NUMANODE);
  if (numa_depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    std::cout << comm->get_rank_in_world() <<
              ": cannot determine hwloc NUMA-node depth" << std::endl;
  }
  int num_numa_nodes = hwloc_get_nbobjs_by_depth(topo, numa_depth);
  // Warn if there are more NUMA nodes than processes per node.
  // It's probably fine if there are more processes than NUMA nodes for now.
  // We can adjust that later when we better understand the threaded perf.
  int ppn = comm->get_procs_per_node();
  if (num_numa_nodes > ppn) {
    if (comm->get_rank_in_world() == 0) {
      std::cout << comm->get_rank_in_world() <<
                ": WARNING: node has " << num_numa_nodes <<
                " NUMA nodes but you have " << ppn << " processes per node" <<
                std::endl;
    }
  }
  hwloc_topology_destroy(topo);
#endif

#ifdef LBANN_HAS_SHMEM
  // Initialize SHMEM
  if (arg_parser.get<bool>(LBANN_OPTION_INIT_SHMEM)) {
    int threading_level = SHMEM_THREAD_MULTIPLE;
    int status = shmem_init_thread(threading_level, &threading_level);
    if (status != 0 || threading_level != SHMEM_THREAD_MULTIPLE) {
      LBANN_ERROR("error initializing OpenSHMEM");
    }
  }
#endif // LBANN_HAS_SHMEM
#ifdef LBANN_HAS_NVSHMEM
  if (arg_parser.get<bool>(LBANN_OPTION_INIT_NVSHMEM)) {
    nvshmem::initialize();
  }
#endif // LBANN_HAS_NVSHMEM

#ifdef LBANN_HAS_DISTCONV
  dc::initialize(MPI_COMM_WORLD);
#endif // LBANN_HAS_DISTCONV

  return comm;
}

void lbann::finalize(lbann_comm* comm) {
  finalize_trainer();
#ifdef LBANN_HAS_NVSHMEM
  nvshmem::finalize();
#endif // LBANN_HAS_NVSHMEM
  MPI_Errhandler_free( &err_handle );
#ifdef LBANN_HAS_DISTCONV
  dc::finalize();
#endif
#ifdef LBANN_HAS_DNN_LIB
  dnn_lib::destroy();
#endif
#ifdef LBANN_HAS_EMBEDDED_PYTHON
  python::finalize();
#endif
#ifdef LBANN_HAS_NVSHMEM
  nvshmem::finalize();
#endif // LBANN_HAS_SHMEM
#ifdef LBANN_HAS_SHMEM
  shmem_finalize();
#endif // LBANN_HAS_SHMEM
  if (comm != nullptr) {
    delete comm;
  }
  El::Finalize();
}

auto lbann::data_layout_to_matrix_format(data_layout layout) -> matrix_format
{
  matrix_format format;
  switch(layout) {
  case data_layout::MODEL_PARALLEL:
    format = matrix_format::MC_MR;
    break;
  case data_layout::DATA_PARALLEL:
    /// Weights are stored in STAR_STAR and data in STAR_VC
    format = matrix_format::STAR_STAR;
    break;
  default:
    LBANN_ERROR("Invalid data layout selected");
  }
  return format;
}

auto lbann::to_string(data_layout const& dl) -> std::string
{
  switch (dl) {
  case data_layout::DATA_PARALLEL:
    return "data_parallel";
  case data_layout::MODEL_PARALLEL:
    return "model_parallel";
  case data_layout::invalid:
    return "invalid";
  }
  return "invalid data_layout";
}

auto lbann::data_layout_from_string(std::string const& str) -> data_layout
{
  if (str == "data_parallel" || str == "DATA_PARALLEL")
    return data_layout::DATA_PARALLEL;
  if (str == "model_parallel" || str == "MODEL_PARALLEL")
    return data_layout::MODEL_PARALLEL;
  if (str == "invalid" || str == "INVALID")
    return data_layout::invalid; // Why is this a thing?
  LBANN_ERROR("Unable to convert \"", str, "\" to lbann::data_layout.");
}

auto lbann::to_string(El::Device const& d) -> std::string
{
  switch (d) {
  case El::Device::CPU:
    return "CPU";
#ifdef HYDROGEN_HAVE_GPU
  case El::Device::GPU:
    return "GPU";
#endif // HYDROGEN_HAVE_GPU
  }
  return "invalid El::Device";
}

auto lbann::device_from_string(std::string const& str) -> El::Device
{
  if (str == "cpu" || str == "CPU")
    return El::Device::CPU;
#ifdef HYDROGEN_HAVE_GPU
  if (str == "gpu" || str == "GPU")
    return El::Device::GPU;
#endif
  LBANN_ERROR("Unable to convert \"", str, "\" to El::Device.");
}

auto lbann::to_string(execution_mode m) -> std::string
{
  switch(m) {
  case execution_mode::training:
    return "training";
  case execution_mode::validation:
    return "validation";
  case execution_mode::testing:
    return "testing";
  case execution_mode::prediction:
    return "prediction";
  case execution_mode::tournament:
    return "tournament";
  case execution_mode::inference:
    return "inference";
  case execution_mode::invalid:
    return "invalid";
  default:
      LBANN_ERROR("Invalid execution mode specified");
  }
}

auto lbann::exec_mode_from_string(std::string const& str) -> execution_mode
{
  if (str == "training" || str == "train")
    return execution_mode::training;
  else if (str == "validation" || str == "validate")
      return execution_mode::validation;
  else if (str == "testing" || str == "test")
    return execution_mode::testing;
  else if (str == "prediction" || str == "predict")
    return execution_mode::prediction;
  else if (str == "tournament")
    return execution_mode::tournament;
  else if (str == "invalid")
    return execution_mode::invalid;
  else
    LBANN_ERROR("\"" + str + "\" is not a valid execution mode.");
}

std::istream& operator>>(std::istream& is, lbann::execution_mode& m) {
  std::string tmp;
  is >> tmp;
  m = lbann::exec_mode_from_string(tmp);
  return is;
}

bool lbann::endsWith(const std::string mainStr, const std::string &toMatch)
{
  if(mainStr.size() >= toMatch.size() &&
     mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
    return true;
  else
    return false;
}

void lbann::print_matrix_dims(AbsDistMat *m, const char *name) {
  std::cout << "DISPLAY MATRIX: " << name << " = "
            << m->Height() << " x " << m->Width() << std::endl;
}

void lbann::print_local_matrix_dims(AbsMat *m, const char *name) {
  std::cout << "DISPLAY MATRIX: " << name << " = "
            << m->Height() << " x " << m->Width() << std::endl;
}

void lbann::lbann_mpi_err_handler(MPI_Comm *comm, int *err_code, ... ) {
  char err_string[MPI_MAX_ERROR_STRING];
  int err_string_length;
  MPI_Error_string(*err_code, &err_string[0], &err_string_length);
  LBANN_ERROR("MPI threw this error: ", err_string);
}

// Layers
CEREAL_FORCE_DYNAMIC_INIT(Layer);
CEREAL_FORCE_DYNAMIC_INIT(data_type_layer);

CEREAL_FORCE_DYNAMIC_INIT(argmax_layer);
CEREAL_FORCE_DYNAMIC_INIT(argmin_layer);
CEREAL_FORCE_DYNAMIC_INIT(base_convolution_layer);
CEREAL_FORCE_DYNAMIC_INIT(batch_normalization_layer);
CEREAL_FORCE_DYNAMIC_INIT(batchwise_reduce_sum_layer);
CEREAL_FORCE_DYNAMIC_INIT(bernoulli_layer);
CEREAL_FORCE_DYNAMIC_INIT(bilinear_resize_layer);
CEREAL_FORCE_DYNAMIC_INIT(categorical_accuracy_layer);
CEREAL_FORCE_DYNAMIC_INIT(categorical_random_layer);
CEREAL_FORCE_DYNAMIC_INIT(channelwise_fully_connected_layer);
CEREAL_FORCE_DYNAMIC_INIT(channelwise_mean_layer);
CEREAL_FORCE_DYNAMIC_INIT(channelwise_scale_bias_layer);
CEREAL_FORCE_DYNAMIC_INIT(channelwise_softmax_layer);
CEREAL_FORCE_DYNAMIC_INIT(composite_image_transformation_layer);
CEREAL_FORCE_DYNAMIC_INIT(concatenate_layer);
CEREAL_FORCE_DYNAMIC_INIT(constant_layer);
CEREAL_FORCE_DYNAMIC_INIT(convolution_layer);
CEREAL_FORCE_DYNAMIC_INIT(covariance_layer);
CEREAL_FORCE_DYNAMIC_INIT(crop_layer);
CEREAL_FORCE_DYNAMIC_INIT(cross_entropy_layer);
CEREAL_FORCE_DYNAMIC_INIT(cutout_layer);
CEREAL_FORCE_DYNAMIC_INIT(deconvolution_layer);
CEREAL_FORCE_DYNAMIC_INIT(discrete_random_layer);
CEREAL_FORCE_DYNAMIC_INIT(dropout_layer);
CEREAL_FORCE_DYNAMIC_INIT(dummy_layer);
CEREAL_FORCE_DYNAMIC_INIT(elu_layer);
CEREAL_FORCE_DYNAMIC_INIT(embedding_layer);
CEREAL_FORCE_DYNAMIC_INIT(entrywise_batch_normalization_layer);
CEREAL_FORCE_DYNAMIC_INIT(entrywise_scale_bias_layer);
CEREAL_FORCE_DYNAMIC_INIT(evaluation_layer);
CEREAL_FORCE_DYNAMIC_INIT(fully_connected_layer);
CEREAL_FORCE_DYNAMIC_INIT(gather_layer);
CEREAL_FORCE_DYNAMIC_INIT(gaussian_layer);
CEREAL_FORCE_DYNAMIC_INIT(hadamard_layer);
CEREAL_FORCE_DYNAMIC_INIT(identity_layer);
CEREAL_FORCE_DYNAMIC_INIT(in_top_k_layer);
CEREAL_FORCE_DYNAMIC_INIT(input_layer);
CEREAL_FORCE_DYNAMIC_INIT(instance_norm_layer);
CEREAL_FORCE_DYNAMIC_INIT(l1_norm_layer);
CEREAL_FORCE_DYNAMIC_INIT(l2_norm2_layer);
CEREAL_FORCE_DYNAMIC_INIT(layer_norm_layer);
CEREAL_FORCE_DYNAMIC_INIT(leaky_relu_layer);
CEREAL_FORCE_DYNAMIC_INIT(local_response_normalization_layer);
CEREAL_FORCE_DYNAMIC_INIT(log_softmax_layer);
CEREAL_FORCE_DYNAMIC_INIT(matmul_layer);
CEREAL_FORCE_DYNAMIC_INIT(mean_absolute_error_layer);
CEREAL_FORCE_DYNAMIC_INIT(mean_squared_error_layer);
CEREAL_FORCE_DYNAMIC_INIT(mini_batch_index_layer);
CEREAL_FORCE_DYNAMIC_INIT(mini_batch_size_layer);
CEREAL_FORCE_DYNAMIC_INIT(one_hot_layer);
CEREAL_FORCE_DYNAMIC_INIT(pooling_layer);
CEREAL_FORCE_DYNAMIC_INIT(reduction_layer);
CEREAL_FORCE_DYNAMIC_INIT(relu_layer);
CEREAL_FORCE_DYNAMIC_INIT(reshape_layer);
CEREAL_FORCE_DYNAMIC_INIT(rotation_layer);
CEREAL_FORCE_DYNAMIC_INIT(rowwise_weights_norms_layer);
CEREAL_FORCE_DYNAMIC_INIT(scatter_layer);
CEREAL_FORCE_DYNAMIC_INIT(selu_dropout);
CEREAL_FORCE_DYNAMIC_INIT(slice_layer);
CEREAL_FORCE_DYNAMIC_INIT(softmax_layer);
CEREAL_FORCE_DYNAMIC_INIT(sort_layer);
CEREAL_FORCE_DYNAMIC_INIT(split_layer);
CEREAL_FORCE_DYNAMIC_INIT(stop_gradient_layer);
CEREAL_FORCE_DYNAMIC_INIT(sum_layer);
CEREAL_FORCE_DYNAMIC_INIT(tessellate_layer);
CEREAL_FORCE_DYNAMIC_INIT(top_k_categorical_accuracy_layer);
CEREAL_FORCE_DYNAMIC_INIT(uniform_layer);
CEREAL_FORCE_DYNAMIC_INIT(unpooling_layer);
CEREAL_FORCE_DYNAMIC_INIT(variance_layer);
CEREAL_FORCE_DYNAMIC_INIT(weighted_sum_layer);
CEREAL_FORCE_DYNAMIC_INIT(weights_layer);

CEREAL_FORCE_DYNAMIC_INIT(OperatorLayer);

// "Special" layers
#ifdef LBANN_HAS_FFTW
CEREAL_FORCE_DYNAMIC_INIT(dft_abs_layer);
#endif
#if defined(LBANN_HAS_SHMEM) || defined(LBANN_HAS_NVSHMEM)
CEREAL_FORCE_DYNAMIC_INIT(dist_embedding_layer);
#endif
#ifdef LBANN_HAS_GPU
CEREAL_FORCE_DYNAMIC_INIT(uniform_hash_layer);
#endif

#if defined LBANN_GRU_LAYER_CUDNN_SUPPORTED || defined LBANN_GRU_LAYER_ONEDNN_CPU_SUPPORTED
CEREAL_FORCE_DYNAMIC_INIT(gru_layer);
#endif

// Operators
CEREAL_FORCE_DYNAMIC_INIT(AbsOperator);
CEREAL_FORCE_DYNAMIC_INIT(AcosOperator);
CEREAL_FORCE_DYNAMIC_INIT(AcoshOperator);
CEREAL_FORCE_DYNAMIC_INIT(AddConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(AddOperator);
CEREAL_FORCE_DYNAMIC_INIT(AsinOperator);
CEREAL_FORCE_DYNAMIC_INIT(AsinhOperator);
CEREAL_FORCE_DYNAMIC_INIT(AtanOperator);
CEREAL_FORCE_DYNAMIC_INIT(AtanhOperator);
CEREAL_FORCE_DYNAMIC_INIT(BinaryCrossEntropyOperator);
CEREAL_FORCE_DYNAMIC_INIT(BooleanAccuracyOperator);
CEREAL_FORCE_DYNAMIC_INIT(BooleanFalseNegativeOperator);
CEREAL_FORCE_DYNAMIC_INIT(BooleanFalsePositiveOperator);
CEREAL_FORCE_DYNAMIC_INIT(CeilOperator);
CEREAL_FORCE_DYNAMIC_INIT(ClampOperator);
CEREAL_FORCE_DYNAMIC_INIT(ConstantSubtractOperator);
CEREAL_FORCE_DYNAMIC_INIT(CosOperator);
CEREAL_FORCE_DYNAMIC_INIT(CoshOperator);
CEREAL_FORCE_DYNAMIC_INIT(DivideOperator);
CEREAL_FORCE_DYNAMIC_INIT(EqualConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(EqualOperator);
CEREAL_FORCE_DYNAMIC_INIT(ErfInvOperator);
CEREAL_FORCE_DYNAMIC_INIT(ErfOperator);
CEREAL_FORCE_DYNAMIC_INIT(ExpOperator);
CEREAL_FORCE_DYNAMIC_INIT(Expm1Operator);
CEREAL_FORCE_DYNAMIC_INIT(FloorOperator);
CEREAL_FORCE_DYNAMIC_INIT(GreaterConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(GreaterEqualConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(GreaterEqualOperator);
CEREAL_FORCE_DYNAMIC_INIT(GreaterOperator);
CEREAL_FORCE_DYNAMIC_INIT(LessConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(LessEqualConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(LessEqualOperator);
CEREAL_FORCE_DYNAMIC_INIT(LessOperator);
CEREAL_FORCE_DYNAMIC_INIT(Log1pOperator);
CEREAL_FORCE_DYNAMIC_INIT(LogOperator);
CEREAL_FORCE_DYNAMIC_INIT(LogSigmoidOperator);
CEREAL_FORCE_DYNAMIC_INIT(LogicalAndOperator);
CEREAL_FORCE_DYNAMIC_INIT(LogicalNotOperator);
CEREAL_FORCE_DYNAMIC_INIT(LogicalOrOperator);
CEREAL_FORCE_DYNAMIC_INIT(LogicalXorOperator);
CEREAL_FORCE_DYNAMIC_INIT(MaxConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(MaxOperator);
CEREAL_FORCE_DYNAMIC_INIT(MinConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(MinOperator);
CEREAL_FORCE_DYNAMIC_INIT(ModOperator);
CEREAL_FORCE_DYNAMIC_INIT(MultiplyOperator);
CEREAL_FORCE_DYNAMIC_INIT(NegativeOperator);
CEREAL_FORCE_DYNAMIC_INIT(NotEqualConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(NotEqualOperator);
CEREAL_FORCE_DYNAMIC_INIT(PowOperator);
CEREAL_FORCE_DYNAMIC_INIT(ReciprocalOperator);
CEREAL_FORCE_DYNAMIC_INIT(RoundOperator);
CEREAL_FORCE_DYNAMIC_INIT(RsqrtOperator);
CEREAL_FORCE_DYNAMIC_INIT(SafeDivideOperator);
CEREAL_FORCE_DYNAMIC_INIT(SafeReciprocalOperator);
CEREAL_FORCE_DYNAMIC_INIT(ScaleOperator);
CEREAL_FORCE_DYNAMIC_INIT(SeluOperator);
CEREAL_FORCE_DYNAMIC_INIT(SigmoidBinaryCrossEntropyOperator);
CEREAL_FORCE_DYNAMIC_INIT(SigmoidOperator);
CEREAL_FORCE_DYNAMIC_INIT(SignOperator);
CEREAL_FORCE_DYNAMIC_INIT(SinOperator);
CEREAL_FORCE_DYNAMIC_INIT(SinhOperator);
CEREAL_FORCE_DYNAMIC_INIT(SoftplusOperator);
CEREAL_FORCE_DYNAMIC_INIT(SoftsignOperator);
CEREAL_FORCE_DYNAMIC_INIT(SqrtOperator);
CEREAL_FORCE_DYNAMIC_INIT(SquareOperator);
CEREAL_FORCE_DYNAMIC_INIT(SquaredDifferenceOperator);
CEREAL_FORCE_DYNAMIC_INIT(SubtractConstantOperator);
CEREAL_FORCE_DYNAMIC_INIT(SubtractOperator);
CEREAL_FORCE_DYNAMIC_INIT(TanOperator);
CEREAL_FORCE_DYNAMIC_INIT(TanhOperator);

// Utilities; miscellaneous
CEREAL_FORCE_DYNAMIC_INIT(SerialMatrixTypes);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_CIRC_CIRC);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_MC_MR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_MC_STAR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_MD_STAR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_MR_MC);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_MR_STAR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_STAR_MC);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_STAR_MD);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_STAR_MR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_STAR_STAR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_STAR_VC);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_STAR_VR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_VC_STAR);
CEREAL_FORCE_DYNAMIC_INIT(DistMat_VR_STAR);

// Callbacks
CEREAL_FORCE_DYNAMIC_INIT(callback_base);
CEREAL_FORCE_DYNAMIC_INIT(callback_check_dataset);
CEREAL_FORCE_DYNAMIC_INIT(callback_check_gradients);
CEREAL_FORCE_DYNAMIC_INIT(callback_check_init);
CEREAL_FORCE_DYNAMIC_INIT(callback_check_metric);
CEREAL_FORCE_DYNAMIC_INIT(callback_check_nan);
CEREAL_FORCE_DYNAMIC_INIT(callback_check_small);
CEREAL_FORCE_DYNAMIC_INIT(callback_compute_model_size);
CEREAL_FORCE_DYNAMIC_INIT(callback_debug);
CEREAL_FORCE_DYNAMIC_INIT(callback_debug_io);
CEREAL_FORCE_DYNAMIC_INIT(callback_dump_error_signals);
CEREAL_FORCE_DYNAMIC_INIT(callback_dump_gradients);
CEREAL_FORCE_DYNAMIC_INIT(callback_dump_minibatch_sample_indices);
CEREAL_FORCE_DYNAMIC_INIT(callback_dump_outputs);
CEREAL_FORCE_DYNAMIC_INIT(callback_dump_weights);
CEREAL_FORCE_DYNAMIC_INIT(callback_early_stopping);
CEREAL_FORCE_DYNAMIC_INIT(callback_gpu_memory_usage);
CEREAL_FORCE_DYNAMIC_INIT(callback_hang);
CEREAL_FORCE_DYNAMIC_INIT(callback_load_model);
CEREAL_FORCE_DYNAMIC_INIT(callback_mixup);
CEREAL_FORCE_DYNAMIC_INIT(callback_monitor_io);
CEREAL_FORCE_DYNAMIC_INIT(callback_perturb_adam);
CEREAL_FORCE_DYNAMIC_INIT(callback_perturb_dropout);
CEREAL_FORCE_DYNAMIC_INIT(callback_perturb_learning_rate);
CEREAL_FORCE_DYNAMIC_INIT(callback_perturb_weights);
CEREAL_FORCE_DYNAMIC_INIT(callback_print_model_description);
CEREAL_FORCE_DYNAMIC_INIT(callback_print_statistics);
CEREAL_FORCE_DYNAMIC_INIT(callback_profiler);
CEREAL_FORCE_DYNAMIC_INIT(callback_save_images);
CEREAL_FORCE_DYNAMIC_INIT(callback_set_weights_value);
CEREAL_FORCE_DYNAMIC_INIT(callback_sync_layers);
CEREAL_FORCE_DYNAMIC_INIT(callback_timeline);
CEREAL_FORCE_DYNAMIC_INIT(callback_timer);

// Optimizers
CEREAL_FORCE_DYNAMIC_INIT(optimizer);
CEREAL_FORCE_DYNAMIC_INIT(data_type_optimizer);

CEREAL_FORCE_DYNAMIC_INIT(adagrad);
CEREAL_FORCE_DYNAMIC_INIT(adam);
CEREAL_FORCE_DYNAMIC_INIT(hypergradient_adam);
CEREAL_FORCE_DYNAMIC_INIT(rmsprop);
CEREAL_FORCE_DYNAMIC_INIT(sgd);

// Other sundries
CEREAL_FORCE_DYNAMIC_INIT(ExecutionContext);
CEREAL_FORCE_DYNAMIC_INIT(SGDExecutionContext);
CEREAL_FORCE_DYNAMIC_INIT(data_coordinator);
CEREAL_FORCE_DYNAMIC_INIT(data_type_weights);
CEREAL_FORCE_DYNAMIC_INIT(dataset);
CEREAL_FORCE_DYNAMIC_INIT(l2_weight_regularization);
CEREAL_FORCE_DYNAMIC_INIT(layer_metric);
CEREAL_FORCE_DYNAMIC_INIT(layer_term);
CEREAL_FORCE_DYNAMIC_INIT(metric);
CEREAL_FORCE_DYNAMIC_INIT(metric_statistics);
CEREAL_FORCE_DYNAMIC_INIT(model);
CEREAL_FORCE_DYNAMIC_INIT(objective_function);
CEREAL_FORCE_DYNAMIC_INIT(objective_function_term);
CEREAL_FORCE_DYNAMIC_INIT(weights);
