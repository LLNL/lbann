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
// lbann_data_reader_imagenet .hpp .cpp - DataReader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_IMAGENET_HPP
#define LBANN_DATA_READER_IMAGENET_HPP

#include "lbann_data_reader.hpp"

namespace lbann
{
	class DataReader_ImageNet : public DataReader
	{
	public:
    DataReader_ImageNet(int batchSize, bool shuffle = true);
    DataReader_ImageNet(const DataReader_ImageNet& source);
		~DataReader_ImageNet();

    int fetch_data(Mat& X);
    int fetch_label(Mat& Y);

    /** returns a vector of 256*256*3 vectors; if max_to_process > 0, only
     *  returns that number of inner vectors; this is probably only useful
     *  for development and testing
     */
    int fetch_data(std::vector<std::vector<unsigned char> > &data, size_t max_to_process = 0);

		int get_num_labels() { return m_num_labels; }

		// ImageNet specific functions
    //		bool load(std::string FileDir, std::string ImageFile, std::string LabelFile);
    bool load(std::string imageDir, std::string imageListFile);
    bool load(std::string imageDir, std::string imageListFile, size_t max_sample_count, bool firstN=false);
    bool load(std::string imageDir, std::string imageListFile, double validation_percent, bool firstN=false);
    void free();

		int get_image_width() { return m_image_width; }
		int get_image_height() { return m_image_height; }
		int get_image_depth() { return m_image_depth; }
    int get_linearized_data_size() { return m_image_width * m_image_height * m_image_depth; }
    int get_linearized_label_size() { return m_num_labels; }

    DataReader_ImageNet& operator=(const DataReader_ImageNet& source);

    /* loads pre-computed data for mean subtraction;
     * throws exceptions is something goes wrong
     */
    void load_mean_data(std::string fn);

	private:
		std::string							m_image_dir; // where images are stored
		std::vector<std::pair<std::string, int> > 	ImageList; // list of image files and labels
		int 										m_image_width; // image width (256)
		int 										m_image_height; // image height (256)
		int 										m_image_depth; // image depth (depth)
		int											m_num_labels; // # labels (1000)
    unsigned char *         m_pixels;
    std::vector<unsigned char> m_mean_data;
	};
}

#endif // LBANN_DATA_READER_IMAGENET_HPP
