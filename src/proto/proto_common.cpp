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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/proto/proto_common.hpp"

#include "lbann/base.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/lbann.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/proto/init_image_data_readers.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/file_utils.hpp"
#if defined(LBANN_HAS_CALIPER)
#include "lbann/utils/profiling.hpp"
#endif

#include "lbann/data_ingestion/readers/data_reader_HDF5.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/reader.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <functional>
#include <memory>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>

namespace lbann {

void init_data_readers(
  lbann_comm* comm,
  const lbann_data::LbannPB& p,
  std::map<execution_mode, generic_data_reader*>& data_readers)
{
  const bool master = comm->am_world_master();
  std::ostringstream err;

  const lbann_data::DataReader& d_reader = p.data_reader();
  const int size = d_reader.reader_size();

  const lbann_data::DataSetMetaData& pb_metadata = p.data_set_metadata();

  // A separate explicit validation/tournament set is created only if a reader
  // with role "validate/tournament" is found in the list of data readers.
  // Otherwise, a validation set is created as a fraction of data from the
  // train set.
  bool separate_validation = false;
  bool separate_tournament = false;
  for (int j = 0; j < size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);
    if (readme.role() == "validate") {
      separate_validation = true;
      continue;
    }
    if (readme.role() == "tournament") {
      separate_tournament = true;
      continue;
    }
  }

  for (int j = 0; j < size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);

    const std::string& name = readme.name();

    const bool shuffle = readme.shuffle();

    generic_data_reader* reader = nullptr;

    // This is a hack that should be fixed when we clean up data reader setup.
    bool set_transform_pipeline = true;

    if ((name == "mnist") || (name == "cifar10")) {
      init_org_image_data_reader(readme, master, reader);
      set_transform_pipeline = false;
    }
    else if ((name == "imagenet")) {
      init_image_data_reader(readme, pb_metadata, master, reader);
      reader->set_data_sample_list(readme.sample_list());
      reader->keep_sample_order(readme.sample_list_keep_order());
      set_transform_pipeline = false;
    }
    else if (name == "jag_conduit") {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_transform_pipeline = false;
      auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader);
      const lbann_data::Model& pb_model = p.model();
      reader->set_data_sample_list(readme.sample_list());
      reader_jag_conduit->set_list_per_trainer(
        readme.sample_list_per_trainer());
      reader_jag_conduit->set_list_per_model(readme.sample_list_per_model());
      reader_jag_conduit->keep_sample_order(readme.sample_list_keep_order());

      for (int i = 0; i < pb_model.layer_size(); ++i) {
        const auto& proto_layer = pb_model.layer(i);
        if (proto_layer.has_input()) {
          reader_jag_conduit->set_local_id(readme.role());
          break;
        }
      }
    }
    else if (name == "jag_conduit_hdf5") {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_transform_pipeline = false;
    }
    else if (name == "nci") {
      reader = new data_reader_nci(shuffle);
    }
    else if (name == "smiles") {
      smiles_data_reader* smiles = new smiles_data_reader(shuffle);
      if (readme.label_filename() != "") {
        LBANN_ERROR("Unsupported data reader field label_filename = ",
                    readme.label_filename());
      }
      if (readme.metadata_filename().empty()) {
        LBANN_ERROR(
          "Required SMILES data reader field metadata_filename is missing");
      }
      smiles->set_metadata_filename(readme.metadata_filename());
      reader = smiles;
      reader->set_data_sample_list(readme.sample_list());
    }
    else if (name == "hdf5_data_reader") {
      hdf5_data_reader* dr = new hdf5_data_reader(shuffle);
      dr->keep_sample_order(readme.sample_list_keep_order());
      dr->set_experiment_schema_filename(readme.experiment_schema_filename());
      dr->set_data_schema_filename(readme.data_schema_filename());
      dr->set_has_labels(readme.enable_labels());
      dr->set_has_responses(readme.enable_responses());
      reader = dr;
      reader->set_data_sample_list(readme.sample_list());
    }
    else if (name == "ras_lipid") {
#ifdef LBANN_HAS_CNPY
      auto* ras_lipid = new ras_lipid_conduit_data_reader(shuffle);
      ras_lipid->set_num_labels(readme.num_labels());
      reader = ras_lipid;
#else
      LBANN_ERROR("attempted to construct ras_lipid numpy data reader, "
                  "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
    }
    else if (name == "csv") {
      auto* reader_csv = new csv_reader(shuffle);
      reader_csv->set_label_col(readme.label_col());
      reader_csv->set_response_col(readme.response_col());
      reader_csv->disable_labels(readme.disable_labels());
      reader_csv->enable_responses(readme.disable_responses());
      reader_csv->set_separator(readme.separator()[0]);
      reader_csv->set_skip_cols(readme.skip_cols());
      reader_csv->set_skip_rows(readme.skip_rows());
      reader_csv->set_has_header(readme.has_header());
      reader = reader_csv;
    }
    else if (name == "numpy_npz_conduit_reader") {
#ifdef LBANN_HAS_CNPY
      auto* npz_conduit = new numpy_npz_conduit_reader(shuffle);
      npz_conduit->set_has_labels(!readme.disable_labels());
      npz_conduit->set_has_responses(!readme.disable_responses());
      npz_conduit->set_scaling_factor_int16(readme.scaling_factor_int16());
      if (readme.num_labels() != 0) {
        npz_conduit->set_num_labels(readme.num_labels());
      }
      reader = npz_conduit;
#else
      LBANN_ERROR("attempted to construct numpy_npz_conduit data reader, "
                  "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
    }
    else if (name == "numpy") {
#ifdef LBANN_HAS_CNPY
      auto* reader_numpy = new numpy_reader(shuffle);
      reader_numpy->set_has_labels(!readme.disable_labels());
      reader_numpy->set_has_responses(!readme.disable_responses());
      reader = reader_numpy;
#else
      LBANN_ERROR("attempted to construct numpy data reader, "
                  "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
    }
    else if (name == "numpy_npz") {
#ifdef LBANN_HAS_CNPY
      auto* reader_numpy_npz = new numpy_npz_reader(shuffle);
      reader_numpy_npz->set_has_labels(!readme.disable_labels());
      reader_numpy_npz->set_has_responses(!readme.disable_responses());
      reader_numpy_npz->set_scaling_factor_int16(readme.scaling_factor_int16());
      reader = reader_numpy_npz;
#else
      LBANN_ERROR("attempted to construct numpy_npz data reader, "
                  "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
    }
    else if (name == "cosmoflow_hdf5" || name == "hdf5") {
#ifdef LBANN_HAS_DISTCONV
      if (name == "cosmoflow_hdf5") {
        LBANN_WARNING("The \"cosmoflow_hdf5\" data reader is deprecated. Use "
                      "\"hdf5\" instead.");
      }
      const auto key_data = readme.hdf5_key_data();
      const auto key_labels = readme.hdf5_key_labels();
      const auto key_responses = readme.hdf5_key_responses();
      const auto hyperslab_labels = readme.hdf5_hyperslab_labels();
      auto* reader_hdf5 = new hdf5_reader<DataType>(shuffle,
                                                    key_data,
                                                    key_labels,
                                                    key_responses,
                                                    hyperslab_labels);
      reader_hdf5->set_has_data_field(INPUT_DATA_TYPE_SAMPLES, true);
      reader_hdf5->set_has_data_field(INPUT_DATA_TYPE_LABEL_RECONSTRUCTION,
                                      true);
      reader_hdf5->set_has_labels(!readme.disable_labels());
      reader_hdf5->set_has_responses(!readme.disable_responses());
      reader_hdf5->set_num_responses(readme.num_responses());
      auto filedir = readme.data_filedir();
      if (!endsWith(filedir, "/")) {
        filedir = filedir + "/";
      }
      const auto paths = glob(filedir + readme.data_file_pattern());
      reader_hdf5->set_hdf5_paths(paths);
      reader = reader_hdf5;
#else
      LBANN_ERROR("attempted to construct cosmoflow_hdf5 or hdf5 data reader, "
                  "but LBANN is not built with Distconv");
#endif // LBANN_HAS_DISTCONV
    }
    else if (name == "pilot2_molecular_reader") {
#ifdef LBANN_HAS_CNPY
      pilot2_molecular_reader* reader_pilot2_molecular =
        new pilot2_molecular_reader(readme.num_neighbors(),
                                    readme.max_neighborhood(),
                                    shuffle);
      reader = reader_pilot2_molecular;
#else
      LBANN_ERROR("attempted to construct pilot2_molecular numpy data reader, "
                  "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
    }
    else if (name == "merge_samples" || name == "merge_features" ||
             name == "multi_conduit") {
      // TODO: verify how much of wildcard conflict with label file, label file
      // should be loaded separately
      auto filedir = readme.data_filedir();
      if (!endsWith(filedir, "/")) {
        filedir = filedir + "/";
      }
      auto paths = glob(filedir + readme.data_file_pattern());
      std::vector<generic_data_reader*> npy_readers;
      for (auto i = paths.begin(); i != paths.end(); i++) {
        const auto path = *i;
        if (master) {
          std::cout << "Loading file: " << path << std::endl;
        }
        if (readme.format() == "numpy") {
#ifdef LBANN_HAS_CNPY
          auto* reader_numpy = new numpy_reader(false);
          reader_numpy->set_data_filename(path);
          reader_numpy->set_has_labels(!readme.disable_labels());
          reader_numpy->set_has_responses(!readme.disable_responses());
          npy_readers.push_back(reader_numpy);
#else
          LBANN_ERROR("attempted to construct numpy data reader, "
                      "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
        }
        else if (readme.format() == "numpy_npz") {
#ifdef LBANN_HAS_CNPY
          auto* reader_numpy_npz = new numpy_npz_reader(false);
          reader_numpy_npz->set_data_filename(path);
          reader_numpy_npz->set_has_labels(!readme.disable_labels());
          reader_numpy_npz->set_has_responses(!readme.disable_responses());
          reader_numpy_npz->set_scaling_factor_int16(
            readme.scaling_factor_int16());
          npy_readers.push_back(reader_numpy_npz);
#else
          LBANN_ERROR("attempted to construct numpy data reader, "
                      "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
        }
        else if (readme.format() == "jag_conduit") {
          init_image_data_reader(readme, pb_metadata, master, reader);
          npy_readers.push_back(reader);
        }
        else if (readme.format() == "pilot2_molecular_reader") {
#ifdef LBANN_HAS_CNPY
          pilot2_molecular_reader* reader_pilot2_molecular =
            new pilot2_molecular_reader(readme.num_neighbors(),
                                        readme.max_neighborhood(),
                                        shuffle);
          reader_pilot2_molecular->set_data_filename(path);
          npy_readers.push_back(reader_pilot2_molecular);
#else
          LBANN_ERROR("attempted to construct numpy data reader, "
                      "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
        }
        else if (readme.format() == "csv") {
          auto* reader_csv = new csv_reader(shuffle);
          if (master) {
            std::cout << "Set data filename: " << path << std::endl;
          }
          reader_csv->set_data_filename(path);
          reader_csv->set_label_col(readme.label_col());
          reader_csv->set_response_col(readme.response_col());
          reader_csv->disable_labels(readme.disable_labels());
          reader_csv->enable_responses(readme.disable_responses());
          reader_csv->set_separator(readme.separator()[0]);
          reader_csv->set_skip_cols(readme.skip_cols());
          reader_csv->set_skip_rows(readme.skip_rows());
          reader_csv->set_has_header(readme.has_header());
          reader_csv->set_absolute_sample_count(readme.absolute_sample_count());
          reader_csv->set_use_fraction(readme.fraction_of_data_to_use());
          reader_csv->set_first_n(readme.first_n());
          npy_readers.push_back(reader_csv);
        }
        else {
          err << __FILE__ << " " << __LINE__
              << " :: unknown format for merged data reader: " << name;
          throw lbann_exception(err.str());
        }
      }
      if (name == "merge_samples") {
        data_reader_merge_samples* merged_samples =
          new data_reader_merge_samples(npy_readers, shuffle);
        reader = merged_samples;
      }
      else if (name == "multi_conduit") {
        // note: this is not a mistake! We may have a separate multi_conduit
        //       reader in the future, but for now merge_samples does what we
        //       need.
        data_reader_merge_samples* multi_conduit =
          new data_reader_merge_samples(npy_readers, shuffle);
        reader = multi_conduit;
      }
      else {
        // create label file
        // we can use merge_features without label
        generic_data_reader* label_reader = nullptr;
        if (readme.label_filename() != "") {
          if (master) {
            std::cout << "Set label filename: " << readme.label_filename()
                      << std::endl;
          }
          if (readme.format() == "numpy") {
#ifdef LBANN_HAS_CNPY
            auto* label_numpy = new numpy_reader(false);
            label_numpy->set_label_filename(readme.label_filename());
            label_numpy->set_data_filename(readme.label_filename());
            label_reader = label_numpy;
#else
            LBANN_ERROR("attempted to construct numpy data reader, "
                        "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
          }
          else if (readme.format() ==
                   "csv") { // if format is csv and label_filename is not empty
            auto* label_csv = new csv_reader(shuffle);
            if (master) {
              std::cout << "Set label filename: " << readme.label_filename()
                        << std::endl;
            }
            label_csv->set_label_filename(readme.label_filename());
            label_csv->set_data_filename(readme.label_filename());
            label_csv->disable_labels(readme.disable_labels());
            label_csv->enable_responses(readme.disable_responses());
            label_csv->set_has_header(
              readme.has_header()); // use same as parent file
            label_csv->set_comm(comm);
            label_csv->set_label_col(0); // assume there is only one label file
                                         // and the column and is label column
            label_csv->set_response_col(0);
            label_reader = label_csv;
          }
          else {
            err << __FILE__ << " " << __LINE__
                << " :: unknown format for merged features label: "
                << readme.format();
            throw lbann_exception(err.str());
          }
        }
        // data_reader_merge_features* merged_features = new
        // data_reader_merge_features(npy_readers,label_csv, shuffle);
        data_reader_merge_features* merged_features =
          new data_reader_merge_features(npy_readers, label_reader, shuffle);
        reader = merged_features;
      }
    }
    else if (name == "synthetic") {
      if (readme.num_labels() != 0) {
        reader = new data_reader_synthetic(
          readme.num_samples(),
          parse_list<El::Int>(readme.synth_dimensions()),
          readme.num_labels(),
          shuffle);
      }
      else {
        reader = new data_reader_synthetic(
          readme.num_samples(),
          parse_list<El::Int>(readme.synth_dimensions()),
          parse_list<El::Int>(readme.synth_response_dimensions()),
          shuffle);
      }
    }
    else if (name == "mesh") {
      reader = new mesh_reader(shuffle);
    }
    else if (name == "python") {
#ifdef LBANN_HAS_EMBEDDED_PYTHON
      const auto& params = readme.python();
      reader = new python_reader(params.module(),
                                 params.module_dir(),
                                 params.sample_function(),
                                 params.num_samples_function(),
                                 params.sample_dims_function(),
                                 shuffle);
#else
      LBANN_ERROR("attempted to construct Python data reader, "
                  "but LBANN is not built with Python/C API");
#endif // LBANN_HAS_EMBEDDED_PYTHON
    }
    else if (name == "python_v2") {
#ifdef LBANN_HAS_EMBEDDED_PYTHON
      const auto& params = readme.python_v2();
      reader = new python_reader_v2(params.dataset_path(),
                                    shuffle);
#else
      LBANN_ERROR("attempted to construct Python data reader, "
                  "but LBANN is not built with Python/C API");
#endif // LBANN_HAS_EMBEDDED_PYTHON
    }
    else if (name == "node2vec") {
#ifdef LBANN_HAS_LARGESCALE_NODE2VEC
      const auto& params = readme.node2vec();
      reader = new node2vec_reader(params.graph_file(),
                                   params.epoch_size(),
                                   params.walk_length(),
                                   params.return_param(),
                                   params.inout_param(),
                                   params.num_negative_samples());
#else
      LBANN_ERROR("attempted to construct node2vec data reader, "
                  "but LBANN is not built with "
                  "largescale_node2vec or HavoqGT");
#endif // LBANN_HAS_LARGESCALE_NODE2VEC
    }
    else {
      err << __FILE__ << " " << __LINE__
          << " :: unknown name for data reader: " << name;
      throw lbann_exception(err.str());
    }
    reader->set_comm(comm);

    if (set_transform_pipeline) {
      reader->set_transform_pipeline(
        proto::construct_transform_pipeline(readme));
    }

    if (readme.data_filename() != "") {
      reader->set_data_filename(readme.data_filename());
    }
    if (readme.label_filename() != "" &&
        name !=
          "merge_features") { // label_file set differently for merge_features
      reader->set_label_filename(readme.label_filename());
    }
    if (readme.data_filedir() != "") {
      reader->set_file_dir(readme.data_filedir());
    }
    reader->set_max_files_to_load(readme.max_files_to_load());
    if (readme.data_local_filedir() != "") {
      reader->set_local_file_dir(readme.data_local_filedir());
    }

    reader->set_absolute_sample_count(readme.absolute_sample_count());
    reader->set_use_fraction(readme.fraction_of_data_to_use());
    reader->set_first_n(readme.first_n());

    reader->set_gan_labelling(readme.gan_labelling());
    reader->set_gan_label_value(readme.gan_label_value());

    if (readme.role() == "train") {
      reader->set_role("train");
    }
    else if (readme.role() == "test") {
      reader->set_role("test");
    }
    else if (readme.role() == "validate") {
      reader->set_role("validate");
    }
    else if (readme.role() == "tournament") {
      reader->set_role("tournament");
    }
    else {
      reader->set_role("error");
    }
    if (readme.role() == "train") {
      reader->set_execution_mode_split_fraction(execution_mode::validation,
                                                readme.validation_fraction());
      reader->set_execution_mode_split_fraction(execution_mode::tournament,
                                                readme.tournament_fraction());
    }

    reader->load();

    if (readme.role() == "train") {
      data_readers[execution_mode::training] = reader;
    }
    else if (readme.role() == "test") {
      // While the default validation_fraction is 0.0, this line is added to be
      // consistent with the case of "train"
      reader->set_execution_mode_split_fraction(execution_mode::validation, 0.);
      data_readers[execution_mode::testing] = reader;
    }
    else if (readme.role() == "validate") {
      reader->set_execution_mode_split_fraction(execution_mode::validation, 0.);
      data_readers[execution_mode::validation] = reader;
    }
    else if (readme.role() == "tournament") {
      reader->set_execution_mode_split_fraction(execution_mode::tournament, 0.);
      data_readers[execution_mode::tournament] = reader;
    }

    if (readme.role() == "train") {
      for (auto m : execution_mode_iterator()) {
        if ((m == execution_mode::validation &&
             readme.validation_fraction() > 0. && !separate_validation) ||
            (m == execution_mode::tournament &&
             readme.tournament_fraction() > 0. && !separate_tournament)) {
          generic_data_reader* split_reader = nullptr;

          if (name == "mnist") {
            split_reader = new mnist_reader(shuffle);
            (*(mnist_reader*)split_reader) = (*(mnist_reader*)reader);
          }
          else if (name == "numpy_npz_conduit_reader") {
#ifdef LBANN_HAS_CNPY
            split_reader = new numpy_npz_conduit_reader(
              *dynamic_cast<const numpy_npz_conduit_reader*>(reader));
#else
            LBANN_ERROR("attempted to construct npz_conduit numpy data reader, "
                        "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
          }
          else if (name == "imagenet") {
#ifdef LBANN_HAS_OPENCV
            split_reader = new imagenet_reader(
              *dynamic_cast<const imagenet_reader*>(reader));
#else
            LBANN_ERROR("imagenet reader not supported without OpenCV.");
#endif // LBANN_HAS_OPENCV
          }
          else if (name == "smiles") {
            split_reader = new smiles_data_reader(
              *dynamic_cast<const smiles_data_reader*>(reader));
          }
          else if (name == "jag_conduit") {
            split_reader = new data_reader_jag_conduit(
              *dynamic_cast<const data_reader_jag_conduit*>(reader));
            const std::string role = "validate";
            auto reader_jag_conduit =
              dynamic_cast<data_reader_jag_conduit*>(split_reader);
            reader_jag_conduit->set_role(role);
          }
          else if (name == "ras_lipid") {
#ifdef LBANN_HAS_CNPY
            auto* ras_lipid = new ras_lipid_conduit_data_reader(shuffle);
            ras_lipid->set_num_labels(readme.num_labels());
            split_reader = ras_lipid;
            (*(ras_lipid_conduit_data_reader*)split_reader) =
              (*(ras_lipid_conduit_data_reader*)reader);
#else
            LBANN_ERROR("attempted to construct ras_lipid numpy data reader, "
                        "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
          }
          else if (name == "nci") {
            split_reader = new data_reader_nci(shuffle);
            (*(data_reader_nci*)split_reader) = (*(data_reader_nci*)reader);
          }
          else if (name == "hdf5_data_reader") {
            split_reader = new hdf5_data_reader(shuffle);
            (*(hdf5_data_reader*)split_reader) = (*(hdf5_data_reader*)reader);
          }
          else if (name == "csv") {
            split_reader = new csv_reader(shuffle);
            (*(csv_reader*)split_reader) = (*(csv_reader*)reader);
          }
          else if (name == "numpy") {
#ifdef LBANN_HAS_CNPY
            split_reader = new numpy_reader(shuffle);
            (*(numpy_reader*)split_reader) = (*(numpy_reader*)reader);
#else
            LBANN_ERROR("attempted to construct numpy data reader, "
                        "but LBANN is not built with CNPY");
#endif // LBANN_HAS_CNPY
          }
          else if (name == "merge_samples") {
            split_reader = new data_reader_merge_samples(
              *(data_reader_merge_samples*)reader);
          }
          else if (name == "merge_features") {
            split_reader = new data_reader_merge_features(
              *(data_reader_merge_features*)reader);
          }
          else if (name == "cifar10") {
            split_reader = new cifar10_reader(shuffle);
            (*(cifar10_reader*)split_reader) = (*(cifar10_reader*)reader);
          }
          else if (name == "synthetic") {
            split_reader =
              new data_reader_synthetic(*(data_reader_synthetic*)reader);
            (*(data_reader_synthetic*)split_reader) =
              (*(data_reader_synthetic*)reader);
          }
          else if (name == "mesh") {
            split_reader = new mesh_reader(shuffle);
            (*(mesh_reader*)split_reader) = (*(mesh_reader*)reader);
          }
          else if (name == "python") {
#ifdef LBANN_HAS_EMBEDDED_PYTHON
            const auto& params = readme.python();
            split_reader = new python_reader(params.module(),
                                             params.module_dir(),
                                             params.sample_function(),
                                             params.num_samples_function(),
                                             params.sample_dims_function(),
                                             shuffle);
            (*(python_reader*)split_reader) = (*(python_reader*)reader);
#else
            LBANN_ERROR("attempted to construct Python data reader, "
                        "but LBANN is not built with Python/C API");
#endif // LBANN_HAS_EMBEDDED_PYTHON
          }
          else if (name == "python_v2") {
#ifdef LBANN_HAS_EMBEDDED_PYTHON
            const auto& params = readme.python_v2();
            split_reader = new python_reader_v2(params.dataset_path(),
                                                shuffle);
            (*(python_reader_v2*)split_reader) = (*(python_reader_v2*)reader);
#else
            LBANN_ERROR("attempted to construct Python data reader, "
                        "but LBANN is not built with Python/C API");
#endif // LBANN_HAS_EMBEDDED_PYTHON
          }

          // this will save someone much grief someday:
          if (split_reader == nullptr) {
            LBANN_ERROR("split_reader == nullptr");
          }

          if (m == execution_mode::validation) {
            split_reader->set_role("validate");
          }
          else if (m == execution_mode::tournament) {
            split_reader->set_role("tournament");
          }
          split_reader->use_unused_index_set(m);
          data_store_conduit* store = split_reader->get_data_store_ptr();
          if (store != nullptr) {
            store->set_data_reader_ptr(split_reader);
            split_reader->get_data_store_ptr()->compact_nodes();
          }

          size_t ntrain = reader->get_num_data();
          if (ntrain == 0) {
            LBANN_ERROR("num train samples = 0; something is wrong");
          }

          if (master) {
            size_t num_train = reader->get_num_data();
            size_t num_split = split_reader->get_num_data();
            double validate_fraction =
              ((double)num_split / (double)(num_train + num_split)) * 100.0;
            double train_fraction =
              ((double)num_train / (double)(num_train + num_split)) * 100.0;
            std::cout << "Training using " << train_fraction
                      << "% of the training data set, which is "
                      << reader->get_num_data() << " samples." << std::endl
                      << to_string(m) << " training using " << validate_fraction
                      << "% of the training data set, which is "
                      << split_reader->get_num_data() << " samples.";
            std::cout << std::endl;
          }

          data_readers[m] = split_reader;
        }
      }
    }
  }

  if (master) {
    if (separate_validation) {
      const generic_data_reader* r_train =
        peek_map(data_readers, execution_mode::training);
      const generic_data_reader* r_validate =
        peek_map(data_readers, execution_mode::validation);
      const size_t num_train =
        (r_train == nullptr) ? 0u : r_train->get_num_data();
      const size_t num_validate =
        (r_validate == nullptr) ? 0u : r_validate->get_num_data();
      std::cout << "Training using " << num_train << " samples." << std::endl
                << "Validating using " << num_validate << " samples."
                << std::endl;
    }
    const generic_data_reader* r_test =
      peek_map(data_readers, execution_mode::testing);
    const size_t num_test = (r_test == nullptr) ? 0u : r_test->get_num_data();
    std::cout << "Testing using " << num_test << " samples." << std::endl;
  }
  // remove null data_reader pointers if there is any
  for (auto it = data_readers.cbegin(); it != data_readers.cend();) {
    if (!it->second) {
      it = data_readers.erase(it);
    }
    else {
      ++it;
    }
  }
}

void read_prototext_file(const std::string& fn,
                         lbann_data::LbannPB& pb,
                         const bool /*master*/)
{
  if (fn.rfind("protobin") != std::string::npos)
    lbann::protobuf::load(fn, pb);
  else
    lbann::protobuf::text::load(fn, pb);
}

void read_prototext_string(const std::string& contents,
                           lbann_data::LbannPB& pb,
                           const bool /*master*/)
{
  lbann::protobuf::text::fill(contents, pb);
}

bool write_prototext_file(const std::string& fn, lbann_data::LbannPB& pb)
{
  int fd = open(fn.c_str(), O_APPEND | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    return false;
  }
  auto* output = new google::protobuf::io::FileOutputStream(fd);
  if (!google::protobuf::TextFormat::Print(pb, output)) {
    close(fd);
    delete output;
    return false;
  }
  delete output;
  close(fd);
  return true;
}

void set_data_readers_filenames(const std::string& which,
                                lbann_data::LbannPB& p)
{
  auto& arg_parser = global_argument_parser();
  lbann_data::DataReader* readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j = 0; j < size; j++) {
    lbann_data::Reader* r = readers->mutable_reader(j);
    if (r->role() == which) {
      std::ostringstream s;
      s << "data_filedir_" << which;
      if (arg_parser.get<std::string>(s.str()) != "") {
        r->set_data_filedir(arg_parser.get<std::string>(s.str()));
      }
      else {
        s.clear();
        s.str("");
        s << "data_filedir";
        if (arg_parser.get<std::string>(s.str()) != "") {
          r->set_data_filedir(arg_parser.get<std::string>(s.str()));
        }
      }
      s.clear();
      s.str("");
      s << "data_filename_" << which;
      if (arg_parser.get<std::string>(s.str()) != "") {
        r->set_data_filename(arg_parser.get<std::string>(s.str()));
      }
      s.clear();
      s.str("");
      s << "label_filename_" << which;
      if (arg_parser.get<std::string>(s.str()) != "") {
        r->set_label_filename(arg_parser.get<std::string>(s.str()));
      }
    }
  }
}

void set_data_readers_sample_list(const std::string& which,
                                  lbann_data::LbannPB& p)
{
  auto& arg_parser = global_argument_parser();
  lbann_data::DataReader* readers = p.mutable_data_reader();
  int size = readers->reader_size();
  const std::string key_role = "sample_list_" + which;

  for (int j = 0; j < size; j++) {
    lbann_data::Reader* r = readers->mutable_reader(j);
    if (r->role() == which) {
      r->set_sample_list(arg_parser.get<std::string>(key_role));
    }
  }
}

void set_data_readers_fraction(lbann_data::LbannPB& p)
{
  auto& arg_parser = global_argument_parser();
  double fraction = arg_parser.get<float>(LBANN_OPTION_DATA_READER_FRACTION);
  if (fraction <= 0 || fraction > 1.0) {
    std::ostringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << " --data_reader_fraction=<float> must be > 0 and <= 1.0";
    throw lbann_exception(err.str());
  }
  lbann_data::DataReader* readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j = 0; j < size; j++) {
    lbann_data::Reader* r = readers->mutable_reader(j);
    r->set_fraction_of_data_to_use(fraction);
  }
}

void customize_data_readers_sample_list(const lbann_comm& comm,
                                        lbann_data::LbannPB& p)
{
  lbann_data::DataReader* readers = p.mutable_data_reader();
  const lbann_data::Model& pb_model = p.model();
  int size = readers->reader_size();
  for (int j = 0; j < size; j++) {
    lbann_data::Reader* r = readers->mutable_reader(j);
    std::ostringstream s;
    std::string basename = get_basename_without_ext(r->sample_list());
    std::string ext = get_ext_name(r->sample_list());
    std::string dir = lbann::file::extract_parent_directory(r->sample_list());
    if ((r->sample_list()).empty()) {
      continue;
    }
    if (dir.empty()) {
      dir = ".";
    }

    s << dir << '/';
    if (r->sample_list_per_model()) {
      s << pb_model.name() << "_";
    }
    if (r->sample_list_per_trainer()) {
      s << "t" << comm.get_trainer_rank() << "_";
    }
    s << basename;
    s << "." << ext;
    r->set_sample_list(s.str());
  }
}

void get_cmdline_overrides(const lbann_comm& comm, lbann_data::LbannPB& p)
{
  std::ostringstream err;

  auto& arg_parser = global_argument_parser();
  lbann_data::Trainer* trainer = p.mutable_trainer();
  lbann_data::Model* model = p.mutable_model();
  lbann_data::DataReader* d_reader = p.mutable_data_reader();
  int size = d_reader->reader_size();

  if (arg_parser.get<int>(LBANN_OPTION_ABSOLUTE_SAMPLE_COUNT) != -1) {
    for (int j = 0; j < size; j++) {
      int n = arg_parser.get<int>(LBANN_OPTION_ABSOLUTE_SAMPLE_COUNT);
      lbann_data::Reader* readme = d_reader->mutable_reader(j);
      readme->set_fraction_of_data_to_use(0.0);
      readme->set_absolute_sample_count(n);
    }
  }

  if ((arg_parser.get<std::string>(LBANN_OPTION_DATA_FILEDIR) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_DATA_FILEDIR_TRAIN) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_DATA_FILENAME_TRAIN) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_LABEL_FILENAME_TRAIN) != "")) {
    set_data_readers_filenames("train", p);
  }
  if ((arg_parser.get<std::string>(LBANN_OPTION_DATA_FILEDIR) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_DATA_FILEDIR_VALIDATE) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_DATA_FILENAME_VALIDATE) !=
       "") or
      (arg_parser.get<std::string>(LBANN_OPTION_LABEL_FILENAME_VALIDATE) !=
       "")) {
    set_data_readers_filenames("validate", p);
  }
  if ((arg_parser.get<std::string>(LBANN_OPTION_DATA_FILEDIR) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_DATA_FILEDIR_TEST) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_DATA_FILENAME_TEST) != "") or
      (arg_parser.get<std::string>(LBANN_OPTION_LABEL_FILENAME_TEST) != "")) {
    set_data_readers_filenames("test", p);
  }
  if (arg_parser.get<std::string>(LBANN_OPTION_SAMPLE_LIST_TRAIN) != "") {
    set_data_readers_sample_list("train", p);
  }
  if (arg_parser.get<std::string>(LBANN_OPTION_SAMPLE_LIST_VALIDATE) != "") {
    set_data_readers_sample_list("validate", p);
  }
  if (arg_parser.get<std::string>(LBANN_OPTION_SAMPLE_LIST_TEST) != "") {
    set_data_readers_sample_list("test", p);
  }
  if (arg_parser.get<float>(LBANN_OPTION_DATA_READER_FRACTION) != -1.0) {
    set_data_readers_fraction(p);
  }
  if (arg_parser.get<int>(LBANN_OPTION_MINI_BATCH_SIZE) != -1) {
    trainer->set_mini_batch_size(
      arg_parser.get<int>(LBANN_OPTION_MINI_BATCH_SIZE));
  }
  if (arg_parser.get<int>(LBANN_OPTION_NUM_EPOCHS) != -1) {
    model->set_num_epochs(arg_parser.get<int>(LBANN_OPTION_NUM_EPOCHS));
  }
  if (arg_parser.get<int>(LBANN_OPTION_HYDROGEN_BLOCK_SIZE) != -1) {
    trainer->set_hydrogen_block_size(
      arg_parser.get<int>(LBANN_OPTION_HYDROGEN_BLOCK_SIZE));
  }
  if (arg_parser.get<bool>(LBANN_OPTION_DISABLE_CUDA)) {
    model->set_disable_cuda(arg_parser.get<bool>(LBANN_OPTION_DISABLE_CUDA));
  }
  if (arg_parser.get<int>(LBANN_OPTION_RANDOM_SEED) != 0) {
    trainer->set_random_seed(arg_parser.get<int>(LBANN_OPTION_RANDOM_SEED));
  }
  if (arg_parser.get<bool>(LBANN_OPTION_SERIALIZE_IO)) {
    trainer->set_serialize_io(arg_parser.get<bool>(LBANN_OPTION_SERIALIZE_IO));
  }
}

void print_parameters(const lbann_comm& comm,
                      lbann_data::LbannPB& p,
                      std::vector<int>& root_random_seeds,
                      std::vector<int>& random_seeds,
                      std::vector<int>& data_seq_random_seeds)
{
  if (!comm.am_world_master()) {
    return;
  }

  lbann_data::Trainer const& t = p.trainer();
  lbann_data::Model const& m = p.model();

#ifdef LBANN_HAS_GPU
  bool const disable_cuda = m.disable_cuda();
#else
  bool const disable_cuda = true;
#endif // LBANN_HAS_GPU
#ifdef LBANN_HAS_CUDNN
  bool const disable_cudnn = disable_cuda;
#else
  bool const disable_cudnn = true;
#endif // LBANN_HAS_CUDNN
#ifdef LBANN_DETERMINISTIC
  bool const enable_determinism = true;
#else
  bool const enable_determinism = false;
#endif // LBANN_DETERMINISTIC
#if defined(LBANN_HAS_CALIPER)
  bool const enable_caliper = is_caliper_initialized();
#else
  bool const enable_caliper = false;
#endif

  std::cout << "\nRunning with these parameters:\n"
            << " General:\n"
            << "  datatype size:              " << sizeof(DataType) << '\n'
            << "  mini_batch_size:            " << t.mini_batch_size() << '\n'
            << "  num_epochs:                 " << m.num_epochs() << '\n'
            << "  hydrogen_block_size:        " << t.hydrogen_block_size()
            << '\n'
            << "  procs_per_trainer:          " << comm.get_procs_per_trainer()
            << '\n'
            << "  serialize_io:               " << t.serialize_io() << '\n'
            << "  caliper:                    "
            << (enable_caliper ? "enabled" : "disabled") << '\n'
            << "  cuda:                       "
            << (disable_cuda ? "disabled" : "enabled") << '\n'
            << "  cudnn:                      "
            << (disable_cudnn ? "disabled" : "enabled") << std::endl;
  auto& arg_parser = global_argument_parser();
  bool const allow_global_statistics =
    arg_parser.get<bool>(LBANN_OPTION_ALLOW_MULTITRAINER_GLOBAL_STATISTICS);
  bool const multitrainer_verbose =
    arg_parser.get<bool>(LBANN_OPTION_MULTITRAINER_VERBOSE);
  int const max_rng_seeds =
    arg_parser.get<int>(LBANN_OPTION_MAX_RNG_SEEDS_DISPLAY);
  std::ostringstream root_rng, rng, data_seq_rng;
  for (size_t i = 0; i < random_seeds.size(); i++) {
    int const trainer_rank = comm.map_world_rank_to_trainer_rank(i);
    int const rank_in_trainer = comm.map_world_rank_to_rank_in_trainer(i);
    if (rank_in_trainer < max_rng_seeds) {
      std::ostringstream id;
      id << "[" << trainer_rank << "][" << rank_in_trainer << "]";
      root_rng << id.str() << "=" << std::setfill('0') << std::setw(10)
               << static_cast<unsigned int>(root_random_seeds[i]) << " ";
      rng << id.str() << "=" << std::setfill('0') << std::setw(10)
          << static_cast<unsigned int>(random_seeds[i]) << " ";
      data_seq_rng << id.str() << "=" << std::setfill('0') << std::setw(10)
                   << static_cast<unsigned int>(data_seq_random_seeds[i])
                   << " ";
    }
    else if (rank_in_trainer == max_rng_seeds) {
      root_rng << "... ";
      rng << "... ";
      data_seq_rng << "... ";
    }
  }
  if (!(allow_global_statistics && multitrainer_verbose) &&
      comm.get_num_trainers() > 1) {
    std::ostringstream msg;
    if (comm.get_num_trainers() == 2) {
      msg << "trainer 1";
    }
    else {
      msg << "trainers 1 to " << comm.get_num_trainers() - 1;
    }
    root_rng << "... (Omitting RNGs from " << msg.str() << ")";
    rng << "... (Omitting RNGs from " << msg.str() << ")";
    data_seq_rng << "... (Omitting RNGs from " << msg.str() << ")";
  }
  std::cout << "  root_random_seed[t][r]:     " << root_rng.str() << '\n'
            << "  random_seed[t][r]:          " << rng.str() << '\n'
            << "  data_seq_random_seed[t][r]: " << data_seq_rng.str() << '\n'
            << "  deterministic_exec:         "
            << (enable_determinism ? "enabled" : "disabled") << '\n'
            << "  data_layout:                " << m.data_layout()
            << "     (only used for metrics)" << std::endl;
}

void save_session(const lbann_comm& comm,
                  const int argc,
                  char* const* argv,
                  lbann_data::LbannPB& p)
{
  if (!comm.am_world_master()) {
    return;
  }

  auto& arg_parser = global_argument_parser();

  // do not write output file for a repeated experiment;
  // may want to revisit this decision later ...
  if (arg_parser.get<std::string>(LBANN_OPTION_PROTOTEXT) != "") {
    return;
  }

  // setup file name
  //  Note: If the file name is not unique, append numbers until it is.
  std::string model_name = p.model().name();
  if (model_name.empty()) {
    model_name = "model";
  };
  std::string file_name = model_name + ".prototext";
  El::Int file_name_index = 1;
  while (std::ifstream(file_name.c_str())) {
    file_name_index++;
    file_name =
      (model_name + "_" + std::to_string(file_name_index) + ".prototext");
  }

  // open output file
  std::ofstream out(file_name.c_str());
  if (!out.is_open()) {
    std::ostringstream err;
    err << "failed to open file (" << file_name << ") for writing";
    LBANN_ERROR(err.str());
  }
  std::cout << std::endl
            << "writing options and prototext to file: " << file_name << "\n\n";

  // output all data
  out << "# cmd line for original experiment:\n#  $ ";
  for (int h = 0; h < argc; h++) {
    out << argv[h] << " ";
  }
  std::string lbann_version("unknown: LBANN_VERSION is not defined");

#ifdef LBANN_VERSION
  lbann_version = LBANN_MAKE_STR(LBANN_VERSION);
#endif

  std::time_t r = std::time(nullptr);
  char* tm = std::ctime(&r);
  size_t fixme = strlen(tm);
  tm[fixme - 1] = 0;
  out << "\n#\n# Experiment conducted at: " << tm
      << "\n#\n#\n# Experiment was run with lbann version: " << lbann_version
      << "\n#\n#\n# To rerun the experiment: \n"
      << "#  $ srun -n" << comm.get_procs_in_world() << " " << argv[0]
      << " --prototext=" << file_name << "\n#\n#\n";

  out << "# Selected SLURM Environment Variables:\n";
  std::vector<std::string> v = {"HOST",
                                "SLURM_NODELIST",
                                "SLURM_NNODES",
                                "SLURM_NTASKS",
                                "SLURM_TASKS_PER_NODE"};
  for (auto& i : v) {
    char* c = std::getenv(i.c_str());
    if (c != nullptr) {
      out << "# " << i << "=" << c << std::endl;
    }
  }
  out << "\n#\n#\n";

  std::string s;
  google::protobuf::TextFormat::PrintToString(p, &s);
  out << s;
  out.close();
}

std::string trim(std::string const& str)
{
  // Short-circuit on the empty string
  if (str.size() == 0)
    return std::string();

  const std::string whitespace = "\f\n\r\t\v ";
  auto first = str.find_first_not_of(whitespace);

  // All characters are whitespace; short-circuit.
  if (first == std::string::npos)
    return std::string();

  auto last = str.find_last_not_of(whitespace);
  return str.substr(first, (last - first) + 1);
}

} // namespace lbann
