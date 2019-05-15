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
// lbann_data_reader .hpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_HPP
#define LBANN_DATA_READER_HPP

#include "lbann/base.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/comm.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/data_readers/image_preprocessor.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include <cassert>
#include <algorithm>
#include <string>
#include <vector>
#include <unistd.h>
#include <unordered_set>


#define NOT_IMPLEMENTED(n) { \
  std::stringstream s; \
  s << "the method " << n << " has not been implemented"; \
  throw lbann_exception(s.str()); }

namespace lbann {

class data_store_conduit;
class model;

/**
 * A data reader manages reading in data in a particular format.
 * This abstract base class manages common functionality. In particular, child
 * classes should implement load and the appropriate subset of fetch_datum,
 * fetch_label, and fetch_response.
 */
class generic_data_reader : public lbann_image_preprocessor {
 public:

 #define JAG_NOOP_VOID if (m_jag_partitioned) { return; }
 #define JAG_NOOP_INT if (m_jag_partitioned) { return 0; }

  /**
   * ctor
   */
  generic_data_reader(bool shuffle = true) :
    m_data_store(nullptr),
    m_comm(nullptr),
    m_mini_batch_size(0), m_current_pos(0),
    m_stride_to_next_mini_batch(0), m_base_offset(0), m_model_offset(0),
    m_sample_stride(1), m_iteration_stride(1),
    m_last_mini_batch_size(0),
    m_stride_to_last_mini_batch(0),
    m_reset_mini_batch_index(0),
    m_loaded_mini_batch_idx(0),
    m_current_mini_batch_idx(0),
    m_num_iterations_per_epoch(0), m_global_mini_batch_size(0),
    m_global_last_mini_batch_size(0),
    m_world_master_mini_batch_adjustment(0),
    m_num_parallel_readers(0), m_rank_in_model(0),
    m_max_files_to_load(0),
    m_file_dir(""), m_data_index_list(""), m_data_fn(""), m_label_fn(""),
    m_shuffle(shuffle), m_absolute_sample_count(0), m_validation_percent(0.0),
    m_use_percent(1.0),
    m_master(false),
    m_gan_labelling(false), //default, not GAN
    m_gan_label_value(0),  //If GAN, default for fake label, discriminator model
    m_is_partitioned(false),
    m_partition_overlap(0),
    m_partition_mode(0),
    m_procs_per_partition(1),
    m_io_thread_pool(nullptr),
    m_jag_partitioned(false),
    m_model(nullptr)
  {}
  generic_data_reader(const generic_data_reader&) = default;
  generic_data_reader& operator=(const generic_data_reader&) = default;

  ~generic_data_reader() override {}
  virtual generic_data_reader* copy() const = 0;

  /// set the comm object
  void set_comm(lbann_comm *comm) {
    m_comm = comm;
    set_master(comm->am_world_master());
  }

  /// returns a (possibly nullptr) to comm
  lbann_comm * get_comm() const {
    return m_comm;
  }

  // These non-virtual methods are used to specify where data is, how much to
  // load, etc.

  /**
   * Set base directory for your data.
   */
  void set_file_dir(std::string s);

  /**
   * Set base directory for your locally cached (e.g, on ssd) data.
   */
  void set_local_file_dir(std::string s);

  /**
   * for some data readers (jag_conduit) we load from multiple files;
   * for testing we want to be able to restrict that number
   */
  void set_max_files_to_load(size_t n) {
    m_max_files_to_load = n;
  }

  /**
   * Returns the base directory for your data.
   * If set_file_dir was not called, returns the empty string
   */
  std::string get_file_dir() const;

  /**
   * Returns the base directory for caching files in local ssd
   * If set_local_file_dir was not called, returns the empty string
   */
  std::string get_local_file_dir() const;

  /**
   * Set the index list for your data (images, etc).
   * The index lists contains an enumeration of all samples in the
   * data set.
   */
  void set_data_index_list(std::string s);

  /**
   * Returns the complete index list for your data set.
   */
  std::string get_data_index_list() const;

  /**
   * Set the filename for your data (images, etc).
   * This may either be a complete filepath, or a subdirectory;
   * see note for set_file_dir(). Also, use this method
   * for cases where the file contains a list of files (e.g, imagenet)
   */
  void set_data_filename(std::string s);

  /**
   * Returns the complete filepath to you data file.
   * See not for set_file_dir()
   */
  std::string get_data_filename() const;

  /**
   * Set the filename for your data (images, etc).
   * This may either be a complete filepath, or a subdirectory;
   * see note for set_file_dir()
   */
  void set_label_filename(std::string s);

  /**
   * Returns the complete filepath to you data file.
   * See not for set_file_dir(). Note: some pipelines (autoencoders)
   * will not make use of this method.
   */
  std::string get_label_filename() const;

  /**
   * If set to false, indices (data samples) are not shuffled
   * default (in ctor) is true.
   */
  void set_shuffle(bool b) { m_shuffle = b; }

  /**
   * Returns true if data samples are shuffled.
   */
  bool is_shuffled() const { return m_shuffle; }

  /**
   * Set shuffled indices; primary use is for testing
   * and reproducibility
   */
  void set_shuffled_indices(const std::vector<int> &indices) {
    m_shuffled_indices = indices;
  }

  /**
   * Returns the shuffled indices; primary use is for testing.
   */
  const std::vector<int> & get_shuffled_indices() const {
    return m_shuffled_indices;
  }

  /**
   * Read the first 'n' samples. If nonzero, this over-rides
   * set_absolute_sample_count, set_use_percent. The intent
   * is to use this for testing. A problem with set_absolute_sample_count
   * and set_use_percent is that the entire data set is read in, then
   * a subset is selected
   */
  void set_first_n(int n);

  /**
   * Sets the absolute number of data samples that will be used for training or
   * testing.
   */
  void set_absolute_sample_count(size_t s);

  /**
   * Set the percentage of the data set to use for training and validation or
   * testing.
   * @param s The percentage used, in the range [0, 1].
   */
  void set_use_percent(double s);

  /**
   * Sets the percentage of the dataset to be used for validation.
   * @param s The percentage used, in the range [0, 1].
   */
  virtual void set_validation_percent(double s);

  /**
   * Set an idenifier for the dataset.
   * The role should be one of "train", "test", or "validate".
   */
  virtual void set_role(std::string role) {
    m_role = role;
    if (options::get()->has_string("jag_partitioned")
        && get_role() == "train") {
      m_jag_partitioned = true;
      if (is_master()) {
        std::cerr << "USING JAG DATA PARTITIONING\n";
      }
    }
  }

  /**
   * Get the role for this dataset.
   */
  std::string get_role() const {
    return m_role;
  }

  /**
   * Load the dataset.
   * Each data reader implementation should implement this to initialize its
   * internal data structures, determine the number of samples and their
   * dimensionality (if needed), and set up and shuffle samples.
   */
  virtual void load() = 0;

  /**
   * Prepare to start processing an epoch of data.
   * If shuffle is true, then shuffle the indices of the data set
   * If the base offset is not specified set it to 0
   * If the stride is not specified set it to batch size
   */
  virtual void setup(int num_io_threads, std::shared_ptr<thread_pool> io_thread_pool);

  /** Return this data_reader's type */
  virtual std::string get_type() const = 0;

  /// Fetch this mini-batch's samples into X.
  virtual int fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched);
  /// Fetch this mini-batch's labels into Y.
  virtual int fetch_labels(CPUMat& Y);
  /// Fetch this mini-batch's responses into Y.
  virtual int fetch_responses(CPUMat& Y);

  /**
   * Save pixels to an image. The implementing data reader is responsible for
   * handling format detection, conversion, etc.
   */
  // TODO: This function needs to go away from here
  void save_image(Mat& pixels, const std::string filename,
                          bool do_scale = true) override {
    NOT_IMPLEMENTED("save_image");
  }
  /**
   * During the network's update phase, the data reader will
   * advanced the current position pointer.  If the pointer wraps
   * around, then reshuffle the data indicies.
   */
  virtual bool update(bool is_active_reader);

  /**
   * This is called at the end of update; it permits data readers to
   * perform actions that are specific to their data sets, for example,
   * data_reader_jag_conduit_hdf5 has the 'primary' data reader
   * bcast its shuffled indices to the other data readers. In general
   * most data readers will probably not overide this method.
   * It may also be called outside of update.
   */

  /// Return the number of labels (classes) in this dataset.
  virtual int get_num_labels() const {
    return 0;
  }
  /// Return the number of responses in this dataset.
  virtual int get_num_responses() const {
    return 1;
  }
  /// Get the linearized size (i.e. number of elements) in a sample.
  virtual int get_linearized_data_size() const {
    return 0;
  }
  /// Get the linearized size (i.e. number of elements) in a label.
  virtual int get_linearized_label_size() const {
    return 0;
  }
  /// Get the linearized size (i.e. number of elements) in a response.
  virtual int get_linearized_response_size() const {
    return 1;
  }
  /// get the linearized size of what is identified by desc.
  virtual int get_linearized_size(const std::string& desc) const {
    if (desc == "data") {
      return get_linearized_data_size();
    } else if (desc == "label") {
      return get_linearized_label_size();
    } else if (desc == "response") {
      return get_linearized_response_size();
    }
    return 0;
  }
  /// Get the dimensions of the data.
  virtual const std::vector<int> get_data_dims() const {
    return std::vector<int>(0);
  }
  /// True if the data reader's current position is valid.
  virtual bool position_valid() const {
    return (m_current_pos < get_num_data());
  }
  /// True if the data reader's current position is not valid but within # ranks per model
  /// of the end of the data set (e.g. it is a rank with no valid data on the last iteration)
  virtual bool position_is_overrun() const {
    int end_pos = (int)m_shuffled_indices.size();
    return (m_current_pos >= end_pos && (m_current_pos - end_pos) < m_comm->get_procs_per_trainer());
  }
  /// True if the data reader is at the start of an epoch.
  bool at_new_epoch() const {
    /// Note that data readers can start at a non-zero index if there
    /// are parallel data readers in a model
    return ((m_loaded_mini_batch_idx == m_reset_mini_batch_index)
            && (m_current_mini_batch_idx == 0));
  }
  /// Set the mini batch size
  void set_mini_batch_size(const int s);
  /// Get the mini batch size
  int get_mini_batch_size() const {
    return m_mini_batch_size;
  }
  /// Get the loaded mini-batch size
  int get_loaded_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_mini_batch_size() const;
  /// Get the current global mini-batch size.
  int get_current_global_mini_batch_size() const;
  /// Get the current mini-batch size.
  int get_current_world_master_mini_batch_adjustment(int model_rank) const;
  /// Return the full mini_batch_size.
  int get_mini_batch_max() const {
    return m_mini_batch_size;
  }
  /// Set the mini batch size across all models (global)
  void set_global_mini_batch_size(const int s) {
    JAG_NOOP_VOID
    m_global_mini_batch_size = s;
  }
  /// Return the mini_batch_size across all models (global)
  int get_global_mini_batch_size() const {
    return m_global_mini_batch_size;
  }
  /// Set the mini batch stride
  void set_stride_to_next_mini_batch(const int s) {
    JAG_NOOP_VOID
    m_stride_to_next_mini_batch = s;
  }
  /// Return the mini batch stride.
  int get_stride_to_next_mini_batch() const {
    return m_stride_to_next_mini_batch;
  }
  /// Set the sample stride
  void set_sample_stride(const int s) {
    JAG_NOOP_VOID
    m_sample_stride = s;
  }
  /// Return the sample stride.
  int get_sample_stride() const {
    return m_sample_stride;
  }
  /// Set the iteration stride
  void set_iteration_stride(const int s) {
    m_iteration_stride = s;
  }
  /// Return the iteration stride.
  int get_iteration_stride() const {
    return m_iteration_stride;
  }
  /// Return the base offset.
  virtual void set_base_offset(const int s) {
    JAG_NOOP_VOID
    m_base_offset = s;
  }
  /// Return the base offset.
  int get_base_offset() const {
    return m_base_offset;
  }
  /// Set the model offset
  void set_model_offset(const int s) {
    JAG_NOOP_VOID
    m_model_offset = s;
  }
  /// Return the model offset.
  int get_model_offset() const {
    return m_model_offset;
  }
  /// Set the last mini batch size
  void set_last_mini_batch_size(const int s) {
    JAG_NOOP_VOID
    m_last_mini_batch_size = s;
  }
  /// Return the last mini batch size
  int get_last_mini_batch_size() const {
    return m_last_mini_batch_size;
  }
  /// Set the last mini batch size across all models (global)
  void set_global_last_mini_batch_size(const int s) {
    JAG_NOOP_VOID
    m_global_last_mini_batch_size = s;
  }
  /// Return the last mini batch size across all models (global)
  int get_global_last_mini_batch_size() const {
    return m_global_last_mini_batch_size;
  }
  /// Set the world master mini batch adjustment (global)
  void set_world_master_mini_batch_adjustment(const int s) {
    JAG_NOOP_VOID
    m_world_master_mini_batch_adjustment = s;
  }
  /// Return the world master mini batch adjustment (global)
  int get_world_master_mini_batch_adjustment() const {
    return m_world_master_mini_batch_adjustment;
  }
  /// Set the last mini batch stride
  void set_stride_to_last_mini_batch(const int s) {
    JAG_NOOP_VOID
    m_stride_to_last_mini_batch = s;
  }
  /// Return the last mini batch stride
  int get_stride_to_last_mini_batch() const {
    return m_stride_to_last_mini_batch;
  }
  /// Set the number of parallel readers per model
  void set_num_parallel_readers(const int s) {
    m_num_parallel_readers = s;
  }
  /// Return the number of parallel readers per model
  int get_num_parallel_readers() const {
    return m_num_parallel_readers;
  }
  /// Set the starting mini-batch index for the epoch
  virtual void set_reset_mini_batch_index(const int s) {
    m_reset_mini_batch_index = s;
  }
  /// Return the starting mini-batch index for the epoch
  int get_reset_mini_batch_index() const {
    return m_reset_mini_batch_index;
  }
  /// Return the current mini-batch index for the epoch
  int get_loaded_mini_batch_index() const {
    return m_loaded_mini_batch_idx;
  }
  /// Return the current mini-batch index for the epoch
  int get_current_mini_batch_index() const {
    return m_current_mini_batch_idx;
  }
  /// Set the current position based on the base and model offsets
  void set_initial_position() {
    m_current_pos = m_base_offset + m_model_offset;
    m_loaded_mini_batch_idx = m_reset_mini_batch_index;
    m_current_mini_batch_idx = 0;
  }
  /// Get the current position in the data reader.
  int get_position() const {
    return m_current_pos;
  }
  /// Get the next position in the data reader.
  int get_next_position() const;
  /// Get a pointer to the start of the shuffled indices.
  int *get_indices() {
    return &m_shuffled_indices[0];
  }
  /// Get the number of samples in this dataset.
  virtual int get_num_data() const {
    return (int)m_shuffled_indices.size();
  }
  /// Get the number of unused samples in this dataset.
  int get_num_unused_data() const {
    return (int)m_unused_indices.size();
  }
  /// Get a pointer to the start of the unused sample indices.
  int *get_unused_data() {
    return &m_unused_indices[0];
  }
  const std::vector<int>& get_unused_indices() {
    return m_unused_indices;
  }
  /// Set the number of iterations in each epoch.
  void set_num_iterations_per_epoch(int num_iterations_per_epoch) {
    m_num_iterations_per_epoch = num_iterations_per_epoch;  /// @todo BVE FIXME merge this with alternate approach
  }
  /// Get the number of iterations in each epoch.
  int get_num_iterations_per_epoch() const {
    return m_num_iterations_per_epoch;  /// @todo BVE FIXME merge this with alternate approach
  }

  /// Return the index of the current iteration step in the epoch (also the mini-batch index)
  int get_current_step_in_epoch() const {
    return  m_current_mini_batch_idx;
  }

  /// only the master may write to cerr or cout; primarily for use in debugging during development
  virtual void set_master(bool m) {
    m_master = m;
  }

  /// only the master may write to cerr or cout; primarily for use in debugging during development
  bool is_master() const {
    return m_master;
  }

  /// Allow the reader to know where it is in the model hierarchy
  virtual void set_rank(int rank) {
    m_rank_in_model = rank;
  }

  /// Allow the reader to know where it is in the model hierarchy
  int get_rank() const {
    return m_rank_in_model;
  }

  /**
   * Select the appropriate subset of data based on settings.
   */
  virtual void select_subset_of_data();

  /// called by select_subset_of_data() if data set is partitioned
  void select_subset_of_data_partitioned();

  /**
   * Replaced the shuffled index set with the unused index set, empying the
   * unused set.
   */
  virtual void use_unused_index_set();

  /// partition the dataset amongst the models
  void set_partitioned(bool is_partitioned=true, double overlap=0.0, int mode=0);

  /// returns true if the data set is partitioned
  bool is_partitioned() const { return m_is_partitioned; }

  /// Does the data reader have a unqiue index list per model
  virtual bool has_list_per_model() const { return false; }
  /// Does the data reader have a unqiue index list per trainer
  virtual bool has_list_per_trainer() const { return false; }


  /** \brief Given directory to store checkpoint files, write state to file and add to number of bytes written */
  bool save_to_checkpoint_shared(persist& p, const char *name);

  /** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
  bool load_from_checkpoint_shared(persist& p, const char *name);

  bool save_to_checkpoint_distributed(persist& p, const char *name);

  /** \brief Given directory to store checkpoint files, read state from file and add to number of bytes read */
  bool load_from_checkpoint_distributed(persist& p, const char *name);

  struct packing_header {
    uint64_t current_pos;
    uint64_t current_mini_batch_idx;
    uint64_t data_size;
  };
  bool pack_scalars(persist& p, const char *name) {
    char fieldname[1024];
    lbann::persist_type persist_value;
    std::string s_name(name);
    if(s_name.compare("data_reader_validation") == 0){
      persist_value = persist_type::validate;
    } else {
       persist_value= persist_type::train;
    }


    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_current_mini_batch_idx);

    int size = m_shuffled_indices.size();
    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) size);

    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.write_uint64(persist_value, fieldname, (uint64_t) m_current_pos);

    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.write_int32_contig(persist_value, fieldname, &m_shuffled_indices[0], (uint64_t) size);

    return true;
  }

  bool unpack_scalars(persist& p, struct packing_header *header, const char *name){
    char fieldname[1024];
    lbann::persist_type persist_value;
    std::string s_name(name);
    if(s_name.compare("data_reader_validation") == 0){
      persist_value = persist_type::validate;
    } else {
       persist_value= persist_type::train;
    }
    // Closest to non checkpoint run only loads m_current_pos

    // record minibatch index
    uint64_t val;

    snprintf(fieldname, sizeof(fieldname), "%s_current_mini_batch_idx", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_current_mini_batch_idx = (int) val;

    snprintf(fieldname, sizeof(fieldname), "%s_data_size", name);
    p.read_uint64(persist_value, fieldname, &val);
    auto size = (int) val;

    // get current position within data
    snprintf(fieldname, sizeof(fieldname), "%s_data_position", name);
    p.read_uint64(persist_value, fieldname, &val);
    m_current_pos = (int) val;
    //resize shuffled index array to hold values
    m_shuffled_indices.resize(size);

     //read list of indices
    snprintf(fieldname, sizeof(fieldname), "%s_data_indices", name);
    p.read_int32_contig(persist_value, fieldname, &m_shuffled_indices[0], (uint64_t) size);

    if(header != nullptr){
      //shuffled data indices array size, used for resize after broadcast. Not unpacked.
      header->data_size = size;
      // all else, unpacked and set in unpack header.
      header->current_pos = m_current_pos;
      header->current_mini_batch_idx = m_current_mini_batch_idx;
    }

  return true;
  }

  void unpack_header(struct packing_header& header){
    m_current_pos = (int) header.current_pos;
    m_current_mini_batch_idx = (int) header.current_mini_batch_idx;
  }

  /// returns a const ref to the data store
  virtual const data_store_conduit& get_data_store() const {
    if (m_data_store == nullptr) {
      LBANN_ERROR("m_data_store is nullptr");
    }
    return *m_data_store;
  }

  data_store_conduit* get_data_store_ptr() const {
    return m_data_store;
  }

  /// sets up a data_store; this is called from build_model_from_prototext()
  /// in utils/lbann_library.cpp. This is a bit awkward: would like to call it
  /// when we instantiate the data_store, but we don;t know the mini_batch_size
  /// until later.
  void setup_data_store(int mini_batch_size);

  void instantiate_data_store(const std::vector<int>& local_list_sizes = std::vector<int>());

  // note: don't want to make this virtual, since then all derived classes
  //       would have to override. But, this should only be called from within
  //       derived classes where it makes sense to do so.
  //       Once the sample_list class and file formats are generalized and
  //       finalized, it should (may?) be possible to code a single
  //       preload_data_store method.
  virtual void preload_data_store() {
    LBANN_ERROR("you should not be here");
  }

  void set_gan_labelling(bool has_gan_labelling) {
     m_gan_labelling = has_gan_labelling;
  }
  void set_gan_label_value(int gan_label_value) { m_gan_label_value = gan_label_value; }

  /// support of data store functionality
  void set_data_store(data_store_conduit *g);

  virtual bool data_store_active() const;

  virtual bool priming_data_store() const;

  void set_model(model *m) { m_model = m; }

  /// experimental; used to ensure all readers for jag_conduit_hdf5
  /// have identical shuffled indices
  virtual void post_update() {}

 protected:

  // For use with conduit when samples are corrupt.
  mutable std::unordered_set<int> m_using_random_node;

  /**
   * Return the absolute number of data samples that will be used for training
   * or testing.
   */
  size_t get_absolute_sample_count() const;

  /**
   * Returns the percent of the dataset to be used for training or testing.
   * If training, this is the total for training and validation. Throws if
   * set_use_percent was not called.
   */
  double get_use_percent() const;

  /**
   * Return the percent of the dataset to be used for validation.
   */
  double get_validation_percent() const;

  data_store_conduit *m_data_store;

  lbann_comm *m_comm;

  virtual bool fetch_data_block(CPUMat& X, El::Int thread_index, El::Int mb_size, El::Matrix<El::Int>& indices_fetched);

  /**
   * Fetch a single sample into a matrix.
   * @param X The matrix to load data into.
   * @param data_id The index of the datum to fetch.
   * @param mb_idx The index within the mini-batch.
   */
  virtual bool fetch_datum(CPUMat& X, int data_id, int mb_idx) {
    NOT_IMPLEMENTED("fetch_dataum");
    return false;
  }

  /**
   * Fetch a single label into a matrix.
   * @param Y The matrix to load data into.
   * @param data_id The index of the datum to fetch.
   * @param mb_idx The index within the mini-batch.
   */
  virtual bool fetch_label(CPUMat& Y, int data_id, int mb_idx) {
    NOT_IMPLEMENTED("fetch_label");
    return false;
  }

  /**
   * Fetch a single response into a matrix.
   * @param Y The matrix to load data into.
   * @param data_id The index of the datum to fetch.
   * @param mb_idx The index within the mini-batch.
   */
  virtual bool fetch_response(CPUMat& Y, int data_id, int mb_idx) {
    NOT_IMPLEMENTED("fetch_response");
    return false;
  }

  /**
   * Called before fetch_datum/label/response to allow initialization.
   */
  virtual void preprocess_data_source(int tid) {};
  /**
   * Called after fetch_datum/label/response to allow initialization.
   */
  virtual void postprocess_data_source(int tid) {};

  /// Shuffle indices (uses the data_seq_generator)
  virtual void shuffle_indices();
  /// Shuffle indices and profide a random number generator
  virtual void shuffle_indices(rng_gen& gen);

  int m_mini_batch_size;
  int m_current_pos;
  /// Batch Stride is typically batch_size, but may be a multiple of batch size if there are multiple readers
  int m_stride_to_next_mini_batch;
  /// If there are multiple instances of the reader,
  /// then it may not reset to zero
  int m_base_offset;
  /// If there are multiple models with multiple instances of the reader,
  /// each model's set of readers may not reset to zero
  /// Provide a set of size, strides, and thresholds to handle the last mini batch of a dataset
  int m_model_offset;
  /// Sample stride is used when a mini-batch is finely interleaved across a DATA_PARALELL
  /// distribution.
  int m_sample_stride;
  /// Stride used by parallel data readers within the model
  int m_iteration_stride;

  std::vector<int> m_shuffled_indices;
  /// Record of the indicies that are not being used for training
  std::vector<int> m_unused_indices;

  int m_last_mini_batch_size;
  int m_stride_to_last_mini_batch;
  /// The index at which this data reader starts its epoch
  int m_reset_mini_batch_index;
  /// The index of the current mini-batch that has been loaded
  int m_loaded_mini_batch_idx;
  /// The index of the current mini-batch that is being processed (train/test/validate)
  int m_current_mini_batch_idx;
  int m_num_iterations_per_epoch; /// How many iterations all readers will execute

  int m_global_mini_batch_size;
  int m_global_last_mini_batch_size;
  int m_world_master_mini_batch_adjustment;

  int m_num_parallel_readers; /// How many parallel readers are being used

  int m_rank_in_model;  /// What is the rank of the data reader within a given model
  size_t m_max_files_to_load;
  std::string m_file_dir;
  std::string m_local_file_dir;
  std::string m_data_index_list;
  std::string m_data_fn;
  std::string m_label_fn;
  bool m_shuffle;
  size_t m_absolute_sample_count;
  double m_validation_percent;
  double m_use_percent;
  int m_first_n;
  std::string m_role;

  bool m_master;

  friend class data_reader_merge_features;
  friend class data_reader_merge_samples;

 protected :
  //var to support GAN
  bool m_gan_labelling; //boolean flag of whether its GAN binary label, default is false
  int m_gan_label_value; //zero(0) or 1 label value for discriminator, default is 0

   /// if true, dataset is partitioned amongst several models,
   /// with options overlap (yeah, I know, if there's overlap its
   /// not technically a partition)
   bool m_is_partitioned;

   /// if m_is_partitioned, this determines the amount of overlap
   /// Has no effect if m_is_partitioned = false
   double m_partition_overlap;

   /// mode = 1: share overlap_percent/2 with left and right nabors
   /// mode = 2: there's a set of overlap indices common to all models
   int m_partition_mode;

   /// only relevant if m_is_partitioned = true.  Currently this is same as
   /// comm->num_models()
   int m_num_partitions;

   /// only relevant if m_is_partitioned = true.  Currently this is same as
   /// comm->get_trainer_rank())
   int m_my_partition;

   /// only relevant if m_is_partitioned = true.  Currently this is same as
   /// comm->get_procs_per_trainer)
   int m_procs_per_partition;

  std::vector<std::vector<char>> m_thread_buffer;

  std::shared_ptr<thread_pool> m_io_thread_pool;

  /// special handling for 1B jag; each reader
  /// owns a unique subset of the data
  bool m_jag_partitioned;

  /// called by fetch_data a single time if m_jag_partitioned = true;
  /// this sets various member variables (num_iterations, m_reset_mini_batch_index,
  /// etc.
  void set_jag_variables(int mb_size);
  model *m_model;
};

template<typename T>
inline void set_minibatch_item(Mat& M, const int mb_idx, const T* const ptr, const size_t count) {
  if ((count > 0u) && (ptr == nullptr)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                          " :: attempt to dereference a nullptr ");
  }
  for (size_t i = 0u; i < count; ++i) {
    M.Set(static_cast<El::Int>(i), static_cast<El::Int>(mb_idx), static_cast<DataType>(ptr[i]));
  }
}

}  // namespace lbann

#endif  // LBANN_DATA_READER_HPP
