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
  // int fetch(lbann::data_reader_synthetic& dr, std::map<lbann::input_data_type, lbann::CPUMat*>& input_buffers, El::Matrix<El::Int>& indices_fetched) {
  //   return dr.fetch(input_buffers, indices_fetched);
  // }
};

// std::vector<float> const samples = {-0.89827055,
//                                     -0.56626886,
//                                     -1.3846669,
//                                     1.3600844,
//                                     -1.9542403,
//                                     -0.70621073,
//                                     -0.74526459,
//                                     0.95250905,
//                                     0.10628668,
//                                     1.1374304,
//                                     0.16106518,
//                                     0.28827614,
//                                     0.020423787,
//                                     -0.54684663,
//                                     1.1501037,
//                                     -1.1680318};

// std::vector<long> const label_indices = {2,
//   8,
//   8,
//   8,
//   9,
//   4,
//   4,
//   4,
//   3,
//   7};

// std::vector<float> const responses = {-1.7044438,
//   -0.12688982,
//   0.81554914,
//   0.84976405,
//   2.0809455,
//   0.62109607,
//   -1.9912087,
//   -3.7694533,
//   1.7465373};

TEST_CASE("Synthetic data reader classification tests",
          "[data_reader][synthetic][classification]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  El::Int num_samples = 7;
  El::Int num_labels = 10;
  std::vector<int> dims = {4, 4};
  lbann::data_reader_synthetic* dr = new lbann::data_reader_synthetic(
          num_samples,
          dims,
          num_labels,
          false);
  DataReaderSyntheticWhiteboxTester white_box_tester;

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));
  std::normal_distribution<lbann::DataType> dist(float(0), float(1));

  SECTION("fetch data and label")
  {
    lbann::CPUMat X;
    X.Resize(dims[0]*dims[1], num_samples);
    lbann::CPUMat Y;
    Y.Resize(num_labels, num_samples);
    El::Zeros_seq(Y, num_labels, num_samples);

    auto io_rng = lbann::set_io_generators_local_index(0);
    for(auto j = 0; j < num_samples; j++) {
      white_box_tester.fetch_datum(*dr, X, 0, j);
      El::Print(X);

      //      El::Zeros_seq(Y, 10, num_labels);
      // for(El::Int i = 0; i < Y.Width(); i++) {
        white_box_tester.fetch_label(*dr, Y, 0, j);
      // }
      El::Print(Y);
    }

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

    // CHECK(X.Height() == samples.size());

    // for(El::Int i = 0; i < X.Height(); i++) {
    //   CHECK(X(i,0) == samples[i]);
    // }

    // CHECK(Y.Width() == label_indices.size());

    // for(El::Int j = 0; j < Y.Width(); j++) {
    //   for(El::Int i = 0; i < Y.Height(); i++) {
    //     if(label_indices[j] == i) {
    //       CHECK(Y(i,j) == 1);
    //     }else {
    //       CHECK(Y(i,j) == 0);
    //     }
    //   }
    // }
  }
}

TEST_CASE("Synthetic data reader regression tests",
          "[data_reader][synthetic][regression]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(42, 1);
  lbann::init_data_seq_random(42);

  El::Int num_samples = 4;
  //  El::Int num_labels = 10;
  std::vector<int> dims = {2, 2};
  std::vector<int> response_dims = {3, 3};
  lbann::data_reader_synthetic* dr = new lbann::data_reader_synthetic(
          num_samples,
          dims,
          response_dims,
          false);
  DataReaderSyntheticWhiteboxTester white_box_tester;

  // Create a local copy of the RNG to check the synthetic data reader
  lbann::fast_rng_gen ref_fast_generator;
  ref_fast_generator.seed(lbann::hash_combine(42, 0));
  std::normal_distribution<lbann::DataType> dist(float(0), float(1));

  SECTION("fetch data and response")
  {
    lbann::CPUMat X;
    X.Resize(dims[0]*dims[1], num_samples);
    lbann::CPUMat Y;
    Y.Resize(response_dims[0]*response_dims[1], num_samples);

    auto io_rng = lbann::set_io_generators_local_index(0);
    //    El::Zeros_seq(X, 10, num_labels);
    for(El::Int i = 0; i < num_samples; i++) {
      white_box_tester.fetch_datum(*dr, X, 0, i);
      El::Print(X);
      // white_box_tester.fetch_response(*dr, Y, 0, i);
      // El::Print(Y);
    }
    //    El::Print(X);

    //    CHECK(X.Width() == Y.Width());

    for(El::Int j = 0; j < num_samples; j++) {
      for(El::Int i = 0; i < X.Height(); i++) {
        CHECK(X(i,j) == dist(ref_fast_generator));
      }
      // for(El::Int i = 0; i < Y.Height(); i++) {
      //   CHECK(Y(i,j) == dist(ref_fast_generator));
      // }
      // auto index = lbann::fast_rand_int(ref_fast_generator, num_labels);
      // std::cout << "Here is the reference value " << index << std::endl;
      // for(El::Int i = 0; i < Y.Height(); i++) {
      //   if(index == i) {
      //     CHECK(Y(i,j) == 1);
      //   }else {
      //     CHECK(Y(i,j) == 0);
      //   }
      // }
    }

    // CHECK(X.Height() == samples.size());

    // for(El::Int i = 0; i < X.Height(); i++) {
    //   CHECK(X(i,0) == samples[i]);
    // }

    // CHECK(Y.Height() == responses.size());

    // for(El::Int i = 0; i < Y.Height(); i++) {
    //   CHECK(Y(i,0) == responses[i]);
    // }
  }
}
