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
// data_reader_image .hpp .cpp - generic data reader class for image dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef IMAGE_DATA_READER_HPP
#define IMAGE_DATA_READER_HPP

#include "data_reader.hpp"
#include "lbann/data_store/data_store_conduit.hpp"

namespace lbann {
class image_data_reader : public generic_data_reader {
 public:
  using img_src_t = std::string;
  using label_t = int;
  using sample_t = std::pair<img_src_t, label_t>;

  image_data_reader(bool shuffle = true);
  image_data_reader(const image_data_reader&);
  image_data_reader(const image_data_reader&, const std::vector<int>& ds_sample_move_list);
  image_data_reader(const image_data_reader&, const std::vector<int>& ds_sample_move_list, std::string role);
  image_data_reader& operator=(const image_data_reader&);

  /** Set up imagenet specific input parameters
   *  If argument is set to 0, then this method does not change the value of
   *  the corresponding parameter. However, width and height can only be both
   *  zero or both non-zero.
   */
  virtual void set_input_params(const int width=0, const int height=0, const int num_ch=0, const int num_labels=0);

  // dataset specific functions
  void load() override;

  void setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool) override;

  int get_num_labels() const override {
    return m_num_labels;
  }
  virtual int get_image_width() const {
    return m_image_width;
  }
  virtual int get_image_height() const {
    return m_image_height;
  }
  virtual int get_image_num_channels() const {
    return m_image_num_channels;
  }
  /// Get the total number of channel values in a sample of image(s).
  int get_linearized_data_size() const override {
    return m_image_linearized_size;
  }
  int get_linearized_label_size() const override {
    return m_num_labels;
  }
  const std::vector<int> get_data_dims() const override {
    return {m_image_num_channels, m_image_height, m_image_width};
  }

  /// Return the sample list of current minibatch
  std::vector<sample_t> get_image_list_of_current_mb() const;

  /// Allow read-only access to the entire sample list
  const std::vector<sample_t>& get_image_list() const {
    return m_image_list;
  }

  /**
   * Returns idx-th sample in the initial loading order.
   * The second argument is only to facilitate overloading, and not to be used by users.
   */
  sample_t get_sample(const size_t idx) const {
    return m_image_list.at(idx);
  }

  void preload_data_store() override; 

 protected:
   void copy_members(const image_data_reader &rhs, const std::vector<int>& ds_sample_move_list = std::vector<int>());

  /// Set the default values for the width, the height, the number of channels, and the number of labels of an image
  virtual void set_defaults();
  bool fetch_label(Mat& Y, int data_id, int mb_idx) override;
  void set_linearized_image_size();

  std::string m_image_dir; ///< where images are stored
  std::vector<sample_t> m_image_list; ///< list of image files and labels
  int m_image_width; ///< image width
  int m_image_height; ///< image height
  int m_image_num_channels; ///< number of image channels
  int m_image_linearized_size; ///< linearized image size
  int m_num_labels; ///< number of labels

  void load_conduit_node_from_file(int data_id, conduit::Node &node);

};

}  // namespace lbann

#endif  // IMAGE_DATA_READER_HPP
