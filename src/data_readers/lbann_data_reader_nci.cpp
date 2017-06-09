////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// lbann_data_reader_nci .hpp .cpp - generic_data_reader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_nci.hpp"
#include <stdio.h>
#include <string>

using namespace std;
using namespace El;



lbann::data_reader_nci::data_reader_nci(int batchSize, bool shuffle)
  : generic_data_reader(batchSize, shuffle) {
  m_num_samples = 0;
  //m_num_samples = -1;
  m_num_features = 0;
  m_num_labels = 2; //@todo fix
}

lbann::data_reader_nci::data_reader_nci(int batchSize)
  : data_reader_nci(batchSize, true) {}

//copy constructor
lbann::data_reader_nci::data_reader_nci(const data_reader_nci& source)
  : generic_data_reader((const generic_data_reader&) source),
    m_num_labels(source.m_num_labels), m_num_samples(source.m_num_samples),
    m_num_features(source.m_num_features),m_labels(source.m_labels),
    m_index_map(source.m_index_map),m_infile(source.m_infile)
{ }

lbann::data_reader_nci::~data_reader_nci() {

}


inline int lbann::data_reader_nci::map_label_2int(const std::string label) {
  if(label == "rs") {
    return 0;
  } else if (label == "nr") {
    return 1;
  }
  //else if (label == "mi") return 3;
  else {
    stringstream err;
    err << endl << __FILE__ << " " << __LINE__
        << "  data_reader_nci::map_label_2int() - Unknown label type : " << label;
    throw lbann_exception(err.str());
  }
}


int lbann::data_reader_nci::fetch_data(Mat& X) {
  if(!generic_data_reader::position_valid()) {
    return 0;
  }

  int current_batch_size = getm_batch_size();
  ifstream ifs(m_infile.c_str());
  if (!ifs) {
    stringstream err;
    err << endl << __FILE__ << " " << __LINE__
        << "  data_reader_nci::fectch_data() - can't open file : " << m_infile;
    throw lbann_exception(err.str());
  }

  string line;
  int n = 0;
  for (n = m_current_pos; n < m_current_pos + current_batch_size; ++n) {
    if (n >= (int)m_shuffled_indices.size()) {
      break;
    }

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];

    if(index == 0) {
      continue;  //skip header
    } else {
      std::getline(ifs.seekg(m_index_map[index-1]+index),line);
    }
    istringstream lstream(line);
    string field;
    int col = 0, f=0;

    while(getline(lstream, field, ' ')) {
      col++;
      //if(col == 4) m_labels[index] = this->map_label_2int(field);
      if(col == 4) {
        m_labels[index] = this->map_label_2int(field);
      }
      if (col > 5) {
        if(field.empty()) {
          field = "0";  //set empty feature field (unit) to zero
        }
        X.Set(f,k,stof(field));
        f++;
      }//end if col > 5
    }// end while loop
  } // end for loop (batch)
  ifs.close();
  return (n - m_current_pos);
}

int lbann::data_reader_nci::fetch_label(Mat& Y) {
  if(!generic_data_reader::position_valid()) {
    return 0;
  }
  int current_batch_size = getm_batch_size();
  int n = 0;
  for (n = m_current_pos; n < m_current_pos + current_batch_size; ++n) {
    if (n >= (int)m_shuffled_indices.size()) {
      break;
    }

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];
    int sample_label = 0;
    if(index == 0) {
      continue;  //skip header
    } else {
      sample_label = m_labels[index];
    }

    Y.Set(sample_label, k, 1);
  }
  return (n - m_current_pos);
}

/*Space separated columns are as follows (in order):
1) Drug Name
2) Cell line name
3) Drug response measurement
4) binary response label (derived from column 3 value)
5) ternary response label (derived from column 3 value and recommend we ignore for now)
6+) features*/

void lbann::data_reader_nci::load() {
  string infile = get_data_filename();
  ifstream ifs(infile.c_str());
  if (!ifs) {
    stringstream err;
    err << endl << __FILE__ << " " << __LINE__
        << "  data_reader_nci::load() - can't open file : " << infile;
    throw lbann_exception(err.str());
  }
  m_infile = infile;
  string line;
  int i;
  double offset =0;
  while(std::getline (ifs, line) ) {
    string field;
    offset = offset + line.length();
    istringstream lstream(line);
    i=0;
    m_num_features = 0;
    while(getline(lstream, field, ' ')) {
      i++;
      if (i > 5) {
        m_num_features++;
      }
    }
    m_index_map[m_num_samples] = offset;
    m_num_samples++;
  }
  ifs.close();
  m_labels.resize(m_num_samples);
  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  for (size_t n = 0; n < m_shuffled_indices.size(); ++n) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}

lbann::data_reader_nci& lbann::data_reader_nci::operator=(const data_reader_nci& source) {

  // check for self-assignment
  if (this == &source) {
    return *this;
  }

  // Call the parent operator= function
  generic_data_reader::operator=(source);


  this->m_num_labels = source.m_num_labels;
  this->m_num_samples = source.m_num_samples;
  this->m_num_features = source.m_num_features;
  this->m_labels = source.m_labels;
  this->m_index_map = source.m_index_map;
  this->m_infile = source.m_infile;

  return *this;
}
