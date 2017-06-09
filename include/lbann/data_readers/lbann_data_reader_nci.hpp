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
// lbann_data_reader_nci .hpp .cpp - DataReader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_NCI_HPP
#define LBANN_DATA_READER_NCI_HPP

#include "lbann_data_reader.hpp"



namespace lbann
{
  //@todo rewrite data_reader class to follow coding convention
  class data_reader_nci : public DataReader
  {
    public:
      data_reader_nci(int batchSize, bool shuffle);
      data_reader_nci(int batchSize);
      data_reader_nci(const data_reader_nci& source); //copy constructor
      data_reader_nci& operator=(const data_reader_nci& source); //assignment operator
      ~data_reader_nci();

      int fetch_data(Mat& X);
      int fetch_label(Mat& Y);
      int getNumLabels() { return m_num_labels; } //@todo; check if used

      void load();

      size_t get_num_samples() {return m_num_samples;}
      size_t get_num_features() {return m_num_features;}
      inline int map_label_2int(const std::string label);

      int get_linearized_data_size() { return m_num_features; }
      int get_linearized_label_size() { return m_num_labels; }

    private:
      //@todo add response mode {binary,ternary, continuous}
      int m_num_labels;  //2 for binary response mode
      size_t m_num_samples; //rows
      size_t m_num_features; //cols
      std::vector<int> m_labels;
      std::map<int,double> m_index_map;
      std::string m_infile; //input file name
  };

}

#endif // LBANN_DATA_READER_NCI_HPP
