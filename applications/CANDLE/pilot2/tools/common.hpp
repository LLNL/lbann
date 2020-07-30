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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef __PILOT2_TOOLS_COMMON_HPP_
#define __PILOT2_TOOLS_COMMON_HPP_


namespace lbann {

const int Num_beads = 184;
const int Dims = 3;
const int Word_size = 4;

const int Num_dist = 16836;
  // 16836 is number of euclid distances
  // for j in range(0, 183):
  //   for k in range(j+1, 184):
  //       t += 1

//=======================================================================
struct xyz {
  xyz() {}
  xyz(float xx, float yy, float zz) : x(xx), y(yy), z(zz) { }

  float x;
  float y;
  float z;

  float dist(const xyz &p) {
    return sqrt( 
             (pow( (x-p.x), 2) 
             + pow( (x-p.x), 2) 
             + pow( (x-p.x), 2))
           );  
  }
  friend std::ostream& operator<<(std::ostream& os, const xyz& p);
};

std::ostream& operator<<(std::ostream& os, const xyz &p) {
  os << p.x << "," << p.y << "," << p.z << " ";
  return os;
}

//=======================================================================

//void testme();

bool sanity_check_npz_file(std::map<std::string, cnpy::NpyArray> &a, const std::string filename) {
  const std::vector<size_t> shape = a["bbs"].shape;
  const float num_samples = static_cast<float>(shape[0]);
  const int word_size = static_cast<int>(a["bbs"].word_size);
  bool is_good = true;
  if (shape[1] != Num_beads || shape[2] != Dims || word_size != Word_size) {
    is_good = false;
    std::stringstream s3;
    for (auto t : shape) { s3 << t << " "; }
    LBANN_WARNING("Bad file: ", filename, " word_size: ", word_size, " dinum_samples: ", num_samples, " shape: ", s3.str());
  }
  return is_good;
}

void read_sample(
  int id, 
  std::vector<float> &data,
  std::vector<float> &z_coordinates,
  std::vector<float> &distances) {

  size_t offset = 2 /* n_frames, n_beads */ + id * (Num_beads + Num_dist);
  z_coordinates.resize(Num_beads);
  for (size_t j=offset; j < offset + Num_beads; j++) {
    z_coordinates[j-offset] = data[j];
  }
  offset += Num_beads;
  for (size_t j = offset; j < offset + Num_dist; j++) {
    if (j >= data.size()) {
      LBANN_ERROR("j >= data.size(); j: ",j, " datalsize: ", data.size(), " offset: ", offset, " Num_beads: ",Num_beads, " Num_dist: ", Num_dist);
    }
    if (j-offset >= distances.size()) {
      LBANN_ERROR("j-offset >= data.size(); j-offset: ", j-offset, " data.size: ", data.size());
    }  
    distances[j-offset] = data[j];
  }
}


} //namespace lbann 

#endif   // __PILOT2_TOOLS_COMMON_HPP_
