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
// lbann_data_reader_mnist .hpp .cpp - DataReader class for MNIST dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_MNIST_HPP
#define LBANN_DATA_READER_MNIST_HPP

#include "lbann_data_reader.hpp"



namespace lbann
{
	class DataReader_MNIST : public DataReader
	{
	public:
    DataReader_MNIST(int batchSize, bool shuffle);
    DataReader_MNIST(int batchSize);
    DataReader_MNIST(const DataReader_MNIST& source);
		~DataReader_MNIST();

    int fetch_data(Mat& X);
    int fetch_label(Mat& Y);
		int getNumLabels() { return NumLabels; }

		// MNIST-specific functions
    bool load(std::string FileDir, std::string ImageFile, std::string LabelFile);
    bool load(std::string FileDir, std::string ImageFile, std::string LabelFile, size_t max_sample_count, bool firstN=false);
    bool load(std::string FileDir, std::string ImageFile, std::string LabelFile, double validation_percent, bool firstN=false);
    bool load(std::string FileDir, std::string ImageFile, std::string LabelFile, int *index_set);
    void free();

		int getImageWidth() { return ImageWidth; }
		int getImageHeight() { return ImageHeight; }
    int get_linearized_data_size() { return ImageWidth * ImageHeight; }
    int get_linearized_label_size() { return NumLabels; }

    DataReader_MNIST& operator=(const DataReader_MNIST& source);

  private:
    void clone_image_data(const DataReader_MNIST& source);

	private:
		std::vector<unsigned char*> 	ImageData;
		int 							ImageWidth;
		int 							ImageHeight;
		int								NumLabels;
	};

}

#endif // LBANN_DATA_READER_MNIST_HPP
