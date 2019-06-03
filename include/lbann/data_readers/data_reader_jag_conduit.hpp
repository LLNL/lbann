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

#ifndef _DATA_READER_JAG_CONDUIT_HPP_
#define _DATA_READER_JAG_CONDUIT_HPP_

#include "lbann_config.hpp" 

#include "lbann/data_readers/data_reader_conduit.hpp"
#include "lbann/data_readers/opencv.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"
#include "hdf5.h"
#include "lbann/data_readers/cv_process.hpp"
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <memory>
#include "lbann/data_readers/sample_list_jag.hpp"

namespace lbann {

class data_store_conduit;


/**
 * Loads JAG simulation parameters and results from hdf5 files using conduit interfaces
 */
template<class Ch_t=float, class Conduit_ch_t=conduit::float32_array, class Scalar_t=double, class Input_t=double, class TimeSeries_t=double>
class data_reader_jag_conduit : public data_reader_conduit {
 public:
  using ch_t = float; ///< jag output image channel type
  using conduit_ch_t = conduit::float32_array; ///< conduit type for ch_t array wrapper
  using scalar_t = double; ///< jag scalar output type
  using input_t = double; ///< jag input parameter type

/*
  data_reader_jag_conduit(const std::shared_ptr<cv_process>& pp, bool shuffle = true)
  : data_reader_conduit(pp, shuffle) {
    set_defaults();
  }
*/


  data_reader_jag_conduit(bool shuffle = true) = delete;
  data_reader_jag_conduit(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_jag_conduit(const data_reader_jag_conduit&);
  data_reader_jag_conduit(const data_reader_jag_conduit&, const std::vector<int>& ds_sample_move_list);
  data_reader_jag_conduit& operator=(const data_reader_jag_conduit&);
  ~data_reader_jag_conduit() override;
  data_reader_jag_conduit* copy() const override { return new data_reader_jag_conduit(*this); }

  std::string get_type() const override {
    return "data_reader_jag_conduit";
  }

#ifndef _JAG_OFFLINE_TOOL_MODE_
#else
  /// See if the image size is consistent with the linearized size
  void check_image_data();
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Return the image simulation output of the i-th sample
  std::vector<cv::Mat> get_cv_images(const size_t i, conduit::Node& sample) const;

  /**
   * Return the images of the i-th sample as an 1-D vector of lbann::DataType
   * There is one image per view, each of which is taken at closest to the bang time.
   */
  std::vector<ch_t> get_images(const size_t i, conduit::Node& sample) const;

  /// Return the scalar simulation output data of the i-th sample
  std::vector<scalar_t> get_scalars(const size_t i, conduit::Node& sample) const;

  /// Return the simulation input parameters of the i-th sample
  std::vector<input_t> get_inputs(const size_t i, conduit::Node& sample) const;

  /// A untiliy function to convert the pointer to image data into an opencv image
  static cv::Mat cast_to_cvMat(const std::pair<size_t, const ch_t*> img,
                               const int height, const int num_ch=1);
 protected:

  virtual void set_defaults();

  virtual void copy_members(const data_reader_jag_conduit& rhs, const std::vector<int>& ds_sample_move_list = std::vector<int>());

  /// Make sure that the keys to choose inputs are valid
  void check_input_keys() override;

  bool fetch(CPUMat& X, int data_id, conduit::Node& sample, int mb_idx, int tid,
             const variable_t vt, const std::string tag);
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& X, int data_id, int mb_idx) override;

#ifndef _JAG_OFFLINE_TOOL_MODE_
  /// See if the image size is consistent with the linearized size
  virtual void check_image_data() override;
#endif // _JAG_OFFLINE_TOOL_MODE_

  /// Obtain image data
  std::vector< std::vector<ch_t> > get_image_data(const size_t i, conduit::Node& sample) const;

};


} // end of namespace lbann

#include "data_reader_jag_conduit_impl.hpp"

#endif //_DATA_READER_JAG_CONDUIT_HPP_
