////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>
#include <lbann.pb.h>

#include <cstdlib>
#include <errno.h>
#include <string.h>

//#include "./data_reader_common_catch2.hpp"
#include "lbann/data_readers/data_reader_synthetic.hpp"
#include "lbann/data_readers/utils/input_data_type.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/threads/thread_utils.hpp"
#include "lbann/utils/hash.hpp"

class DataReaderSyntheticWhiteboxTester
{
public:
  bool fetch_datum(lbann::data_reader_synthetic& dr, lbann::CPUMat& X, int data_id, int mb_idx) {
    return dr.fetch_datum(X, data_id, mb_idx);
  }
  bool fetch_label(lbann::data_reader_synthetic& dr, lbann::CPUMat& Y, int data_id, int mb_idx) {
    return dr.fetch_label(Y, data_id, mb_idx);
  }
  bool fetch_response(lbann::data_reader_synthetic& dr, lbann::CPUMat& Y, int data_id, int mb_idx) {
    return dr.fetch_response(Y, data_id, mb_idx);
  }
};

TEST_CASE("Synthetic data reader public API classification tests",
          "[mpi][data_reader][synthetic][classification][public]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));
  std::normal_distribution<lbann::DataType> dist(float(0), float(1));

  // Initalize a per-trainer I/O thread pool
  auto io_thread_pool = lbann::make_unique<lbann::thread_pool>();
  io_thread_pool->launch_pinned_threads(1, 1);

  El::Int num_samples = 7;
  El::Int num_labels = 10;
  std::vector<int> dims = {4, 4};
  lbann::data_reader_synthetic* dr = new lbann::data_reader_synthetic(
          num_samples,
          dims,
          num_labels,
          false);
  dr->setup(io_thread_pool->get_num_threads(),
            io_thread_pool.get());
  dr->set_rank(0);
  dr->set_comm(&comm);
  dr->set_num_parallel_readers(1);
  dr->load();
  dr->set_mini_batch_size(num_samples);
  dr->set_last_mini_batch_size(num_samples);
  dr->set_initial_position();

  DataReaderSyntheticWhiteboxTester white_box_tester;

  std::vector<std::string> active_data_fields = {"samples", "labels"};
  //  std::vector<std::string> active_data_fields = {"samples", "labels", "responses"};
  std::map<lbann::input_data_type, lbann::CPUMat*> local_input_buffers;
  for (auto& data_field : active_data_fields) {
    lbann::input_data_type data_field_hack;
    lbann::CPUMat* local_mat = new lbann::CPUMat();
    if (data_field == INPUT_DATA_TYPE_SAMPLES) {
      data_field_hack = lbann::input_data_type::SAMPLES;
      local_mat->Resize(dims[0]*dims[1], num_samples);
      El::Zeros_seq(*local_mat, dims[0]*dims[1], num_samples);
    }
    else if (data_field == INPUT_DATA_TYPE_LABELS) {
      data_field_hack = lbann::input_data_type::LABELS;
      local_mat->Resize(num_labels, num_samples);
      El::Zeros_seq(*local_mat, num_labels, num_samples);
    }
    else if (data_field == INPUT_DATA_TYPE_LABELS) {
      data_field_hack = lbann::input_data_type::RESPONSES;
    }
    local_input_buffers[data_field_hack] = local_mat;
  }
  El::Matrix<El::Int> indices_fetched;
  El::Zeros_seq(indices_fetched, num_samples, 1);

  // for(auto k = 1; k < 10; k++) {
  //   dims[0] = k;
  //   for(auto l = 1; l < 10; l++) {
  //     response_dims[0] = l;

      SECTION("fetch data and label")
      {
        dr->fetch(local_input_buffers, indices_fetched);

        for(auto& [field, buf] : local_input_buffers) {
          std::cout << "For field " << to_string(field) << std::endl;
          El::Print(*buf);
        }

        auto& X = *(local_input_buffers[lbann::input_data_type::SAMPLES]);
        auto& Y = *(local_input_buffers[lbann::input_data_type::LABELS]);
        CHECK(X.Width() == Y.Width());

        for(El::Int j = 0; j < Y.Width(); j++) {
          for(El::Int i = 0; i < X.Height(); i++) {
            CHECK(X(i,j) == dist(ref_fast_generator));
          }
          auto index = lbann::fast_rand_int(ref_fast_generator, num_labels);
          std::cout << "Here is the reference value " << index << std::endl;
          for(El::Int i = 0; i < Y.Height(); i++) {
            if(index == i) {
              CHECK(Y(i,j) == 1);
            }else {
              CHECK(Y(i,j) == 0);
            }
          }
        }
      }
  //   }
  // }

  for(auto& [field, buf] : local_input_buffers) {
    delete buf;
  }
}
