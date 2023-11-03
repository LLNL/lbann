////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_readers/sample_list.hpp"
#include "lbann/data_store/data_store_conduit.hpp"

namespace lbann {
class image_data_reader : public generic_data_reader
{
public:
  using img_src_t = std::string;
  using label_t = int;
  using sample_t = std::pair<img_src_t, label_t>;
  using sample_name_t = img_src_t;
  using sample_list_t = sample_list<sample_name_t>;
  using sample_idx_t = sample_list_t::sample_idx_t;
  using labels_t = std::vector<label_t>;

  image_data_reader(bool shuffle = true);
  image_data_reader(const image_data_reader&);
  image_data_reader& operator=(const image_data_reader&);

  /** Set up imagenet specific input parameters
   *  If argument is set to 0, then this method does not change the value of
   *  the corresponding parameter. However, width and height can only be both
   *  zero or both non-zero.
   */
  virtual void set_input_params(const int width = 0,
                                const int height = 0,
                                const int num_ch = 0,
                                const int num_labels = 0);

  // dataset specific functions
  void load() override;

  void setup(int num_io_threads,
             observer_ptr<thread_pool> io_thread_pool) override;

  int get_num_labels() const override { return m_num_labels; }
  virtual int get_image_width() const { return m_image_width; }
  virtual int get_image_height() const { return m_image_height; }
  virtual int get_image_num_channels() const { return m_image_num_channels; }
  /// Get the total number of channel values in a sample of image(s).
  int get_linearized_data_size() const override
  {
    return m_image_linearized_size;
  }
  int get_linearized_label_size() const override { return m_num_labels; }
  const std::vector<El::Int> get_data_dims() const override
  {
    return {m_image_num_channels, m_image_height, m_image_width};
  }

  /// Allow read-only access to the entire sample list
  const sample_list_t& get_sample_list() const { return m_sample_list; }

  /**
   * Returns idx-th sample in the initial loading order.
   * The second argument is only to facilitate overloading, and not to be used
   * by users.
   */
  sample_t get_sample(const size_t idx) const;

  void do_preload_data_store() override;

  void load_conduit_node_from_file(uint64_t data_id, conduit::Node& node);

protected:
  void copy_members(const image_data_reader& rhs);

  /// Set the default values for the width, the height, the number of channels,
  /// and the number of labels of an image
  virtual void set_defaults();
  bool fetch_label(Mat& Y, uint64_t data_id, uint64_t mb_idx) override;
  void set_linearized_image_size();

  /** Dump the image list file in which each line consists of the file name
   *  and the label of a sample */
  void dump_sample_label_list(const std::string& dump_file_name);
  /// Rely on pre-determined list of samples.
  void load_list_of_samples(const std::string filename);
  /// Load the sample list from a serialized archive from another rank
  void
  load_list_of_samples_from_archive(const std::string& sample_list_archive);
  /// Use the imagenet image list file, and generate sample list header
  /// on-the-fly
  void gen_list_of_samples();
  /// Load the labels for samples
  void load_labels(std::vector<char>& preloaded_buffer);
  /// Read the labels from an open input stream
  void read_labels(std::istream& istrm);
  /// Return the number of lines in the input stream
  size_t determine_num_of_samples(std::istream& istrm) const;

  std::string m_image_dir;     ///< where images are stored
  int m_image_width;           ///< image width
  int m_image_height;          ///< image height
  int m_image_num_channels;    ///< number of image channels
  int m_image_linearized_size; ///< linearized image size
  int m_num_labels;            ///< number of labels

  sample_list_t m_sample_list;
  labels_t m_labels;

  bool load_conduit_nodes_from_file(const std::unordered_set<int>& data_ids);
};

} // namespace lbann

#endif // IMAGE_DATA_READER_HPP
