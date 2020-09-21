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

#include "lbann/proto/proto_common.hpp"

#include "lbann/lbann.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/proto/init_image_data_readers.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/argument_parser.hpp"

#include <lbann.pb.h>
#include <reader.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

namespace lbann {

int get_requested_num_parallel_readers(
  const lbann_comm& comm, const lbann_data::LbannPB& p);

void init_data_readers(
  lbann_comm* comm, const lbann_data::LbannPB& p,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  bool is_shareable_training_data_reader,
  bool is_shareable_testing_data_reader,
  bool is_shareable_validation_data_reader)
{
  static std::unordered_map<std::string, data_reader_jag_conduit*> leading_reader_jag_conduit;
  const bool master = comm->am_world_master();
  std::ostringstream err;

  options *opts = options::get();
  const bool create_tarball
    = opts->has_string("create_tarball") ? true : false;

  const lbann_data::DataReader & d_reader = p.data_reader();
  const int size = d_reader.reader_size();

  const lbann_data::DataSetMetaData& pb_metadata = p.data_set_metadata();

  // A separate explicit validation set is created only if a reader with role "validate"
  // is found in the list of data readers. Otherwise, a validation set is created as a
  // percentage of data from the train set.
  bool separate_validation = false;
  for (int j=0; j<size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);
    if (readme.role() == "validate") {
        separate_validation = true;
        break;
    }
  }

  for (int j=0; j<size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);

    const std::string& name = readme.name();

    const bool shuffle = readme.shuffle();

    generic_data_reader *reader = nullptr;
    generic_data_reader *reader_validation = nullptr;

    // This is a hack that should be fixed when we clean up data reader setup.
    bool set_transform_pipeline = true;

    if ((name == "mnist") || (name == "cifar10")) {
      init_org_image_data_reader(readme, master, reader);
      set_transform_pipeline = false;
    } else if ((name == "imagenet")) {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_transform_pipeline = false;
    } else if (name == "jag_conduit") {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_transform_pipeline = false;
      auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader);
      const lbann_data::Model& pb_model = p.model();
      const lbann_data::Trainer& pb_trainer = p.trainer();
      reader->set_mini_batch_size(static_cast<int>(pb_trainer.mini_batch_size()));
      reader->set_data_index_list(readme.index_list());
      reader_jag_conduit->set_list_per_trainer(readme.index_list_per_trainer());
      reader_jag_conduit->set_list_per_model(readme.index_list_per_model());

      /// Allow the prototext to control if the data readers is
      /// shareable for each phase training, validation, or testing
      if((is_shareable_training_data_reader && readme.role() == "train")
         || (is_shareable_testing_data_reader && readme.role() == "test")
         || (is_shareable_validation_data_reader && readme.role() == "validation")) {
        if (!peek_map(leading_reader_jag_conduit, readme.role())) {
          leading_reader_jag_conduit[readme.role()] = reader_jag_conduit;
        } else {
          const auto leader = peek_map(leading_reader_jag_conduit, readme.role());
          *reader_jag_conduit = *leader;
          reader_jag_conduit->set_leading_reader(leader);
        }
      }

      for (int i=0; i < pb_model.layer_size(); ++i) {
        const auto& proto_layer = pb_model.layer(i);
        if (proto_layer.has_input()) {
          const auto& params = proto_layer.input();
          const auto& io_buffer = params.io_buffer();
          reader_jag_conduit->set_io_buffer_type(io_buffer);
          const auto num_readers = get_requested_num_parallel_readers(*comm, p);
          reader_jag_conduit->set_num_parallel_readers(num_readers);
          reader_jag_conduit->set_local_id(readme.role());
          break;
        }
      }
    } else if (name == "jag_conduit_hdf5") {
      init_image_data_reader(readme, pb_metadata, master, reader);
      set_transform_pipeline = false;
    } else if (name == "nci") {
      reader = new data_reader_nci(shuffle);
    } else if (name == "smiles") {
      smiles_data_reader * smiles = new smiles_data_reader(shuffle);
      reader = smiles;
    } else if (name == "ras_lipid") {
      auto *ras_lipid = new ras_lipid_conduit_data_reader(shuffle);
      ras_lipid->set_num_labels(readme.num_labels());
      reader = ras_lipid;
    } else if (name == "csv") {
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
    } else if (name == "numpy_npz_conduit_reader") {
      auto *npz_conduit = new numpy_npz_conduit_reader(shuffle);
      npz_conduit->set_has_labels(!readme.disable_labels());
      npz_conduit->set_has_responses(!readme.disable_responses());
      npz_conduit->set_scaling_factor_int16(readme.scaling_factor_int16());
      if (readme.num_labels() != 0) {
        npz_conduit->set_num_labels(readme.num_labels());
      }
      reader = npz_conduit;
    } else if (name == "numpy") {
      auto* reader_numpy = new numpy_reader(shuffle);
      reader_numpy->set_has_labels(!readme.disable_labels());
      reader_numpy->set_has_responses(!readme.disable_responses());
      reader = reader_numpy;
    } else if (name == "numpy_npz") {
      auto* reader_numpy_npz = new numpy_npz_reader(shuffle);
      reader_numpy_npz->set_has_labels(!readme.disable_labels());
      reader_numpy_npz->set_has_responses(!readme.disable_responses());
      reader_numpy_npz->set_scaling_factor_int16(readme.scaling_factor_int16());
      reader = reader_numpy_npz;
#ifdef LBANN_HAS_DISTCONV
    } else if (name=="cosmoflow_hdf5") {
      auto* reader_cosmo_hdf5 = new hdf5_reader(shuffle);
      auto filedir = readme.data_filedir();
      if(!endsWith(filedir, "/")) {
        filedir = filedir + "/";
      }
      const auto paths = glob(filedir +readme.data_file_pattern());
      reader_cosmo_hdf5->set_hdf5_paths(paths);
      reader = reader_cosmo_hdf5;
#endif // LBANN_HAS_DISTCONV
    } else if (name == "pilot2_molecular_reader") {
      pilot2_molecular_reader* reader_pilot2_molecular = new pilot2_molecular_reader(readme.num_neighbors(), readme.max_neighborhood(), shuffle);
      reader = reader_pilot2_molecular;
    } else if (name == "merge_samples" || name == "merge_features" || name == "multi_conduit") {
      //TODO: verify how much of wildcard conflict with label file, label file should be loaded separately
      auto filedir = readme.data_filedir();
      if(!endsWith(filedir, "/")) {
        filedir = filedir + "/";
      }
      auto paths = glob(filedir + readme.data_file_pattern());
      std::vector<generic_data_reader*> npy_readers;
      for(auto i = paths.begin(); i != paths.end(); i++) {
        const auto path = *i;
        if(master) { std::cout << "Loading file: " << path << std::endl; }
        if (readme.format() == "numpy") {
          auto *reader_numpy = new numpy_reader(false);
          reader_numpy->set_data_filename(path);
          reader_numpy->set_has_labels(!readme.disable_labels());
          reader_numpy->set_has_responses(!readme.disable_responses());
          npy_readers.push_back(reader_numpy);
        } else if (readme.format() == "numpy_npz") {
          auto* reader_numpy_npz = new numpy_npz_reader(false);
          reader_numpy_npz->set_data_filename(path);
          reader_numpy_npz->set_has_labels(!readme.disable_labels());
          reader_numpy_npz->set_has_responses(!readme.disable_responses());
          reader_numpy_npz->set_scaling_factor_int16(readme.scaling_factor_int16());
          npy_readers.push_back(reader_numpy_npz);
        } else if (readme.format() == "jag_conduit") {
          init_image_data_reader(readme, pb_metadata, master, reader);
          npy_readers.push_back(reader);
        } else if (readme.format() == "pilot2_molecular_reader") {
          pilot2_molecular_reader* reader_pilot2_molecular = new pilot2_molecular_reader(readme.num_neighbors(), readme.max_neighborhood(), shuffle);
          reader_pilot2_molecular->set_data_filename(path);
          npy_readers.push_back(reader_pilot2_molecular);
        } else if (readme.format() == "csv") {
          auto* reader_csv = new csv_reader(shuffle);
          if(master) { std::cout << "Set data filename: " << path << std::endl; }
          reader_csv->set_data_filename(path);
          reader_csv->set_label_col(readme.label_col());
          reader_csv->set_response_col(readme.response_col());
          reader_csv->disable_labels(readme.disable_labels());
          reader_csv->enable_responses(readme.disable_responses());
          reader_csv->set_separator(readme.separator()[0]);
          reader_csv->set_skip_cols(readme.skip_cols());
          reader_csv->set_skip_rows(readme.skip_rows());
          reader_csv->set_has_header(readme.has_header());
          reader_csv->set_absolute_sample_count( readme.absolute_sample_count() );
          reader_csv->set_use_percent( readme.percent_of_data_to_use() );
          reader_csv->set_first_n( readme.first_n() );
          npy_readers.push_back(reader_csv);
        } else {
          err << __FILE__ << " " << __LINE__ << " :: unknown format for merged data reader: "
              << name;
          throw lbann_exception(err.str());
        }
      }
      if(name == "merge_samples") {
        data_reader_merge_samples* merged_samples = new data_reader_merge_samples(npy_readers, shuffle);
        reader = merged_samples;
      } else if (name == "multi_conduit") {
        //note: this is not a mistake! We may have a separate multi_conduit
        //      reader in the future, but for now merge_samples does what we need.
        data_reader_merge_samples* multi_conduit = new data_reader_merge_samples(npy_readers, shuffle);
        reader = multi_conduit;
      } else {
        //create label file
        //we can use merge_features without label
        generic_data_reader* label_reader = nullptr;
        if(readme.label_filename() != "") {
          if(master) { std::cout << "Set label filename: " << readme.label_filename() << std::endl; }
          if (readme.format() == "numpy") {
             auto* label_numpy  = new numpy_reader(false);
             label_numpy->set_label_filename(readme.label_filename());
             label_numpy->set_data_filename(readme.label_filename());
             label_reader = label_numpy;
           } else if (readme.format() == "csv") { //if format is csv and label_filename is not empty
             auto* label_csv = new csv_reader(shuffle);
             if(master) { std::cout << "Set label filename: " << readme.label_filename() << std::endl; }
             label_csv->set_label_filename(readme.label_filename());
             label_csv->set_data_filename(readme.label_filename());
             label_csv->disable_labels(readme.disable_labels());
             label_csv->enable_responses(readme.disable_responses());
             label_csv->set_has_header(readme.has_header()); //use same as parent file
             label_csv->set_comm(comm);
             label_csv->set_label_col(0); //assume there is only one label file and the column and is label column
             label_csv->set_response_col(0);
             label_reader = label_csv;
           } else {
             err << __FILE__ << " " << __LINE__ << " :: unknown format for merged features label: "
                << readme.format();
             throw lbann_exception(err.str());
           }
         }
        //data_reader_merge_features* merged_features = new data_reader_merge_features(npy_readers,label_csv, shuffle);
        data_reader_merge_features* merged_features = new data_reader_merge_features(npy_readers,label_reader, shuffle);
        reader = merged_features;
      }

    } else if (name == "synthetic") {
      if (readme.num_labels() != 0) {
        reader = new data_reader_synthetic(
          readme.num_samples(),
          parse_list<int>(readme.synth_dimensions()),
          readme.num_labels(),
          shuffle);
      } else {
        reader = new data_reader_synthetic(
          readme.num_samples(),
          parse_list<int>(readme.synth_dimensions()),
          parse_list<int>(readme.synth_response_dimensions()),
          shuffle);
      }
    } else if (name == "mesh") {
      reader = new mesh_reader(shuffle);
    } else if (name == "python") {
#ifdef LBANN_HAS_PYTHON
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
#endif // LBANN_HAS_PYTHON
    } else {
        err << __FILE__ << " " << __LINE__ << " :: unknown name for data reader: "
            << name;
        throw lbann_exception(err.str());
    }
    reader->set_comm(comm);

    if (set_transform_pipeline) {
      reader->set_transform_pipeline(
        proto::construct_transform_pipeline(readme));
    }

    if (readme.data_filename() != "") {
      reader->set_data_filename( readme.data_filename() );
    }
    if (readme.label_filename() != "" && name != "merge_features") { //label_file set differently for merge_features
      reader->set_label_filename( readme.label_filename() );
    }
    if (readme.data_filedir() != "") {
      reader->set_file_dir( readme.data_filedir() );
    }
    reader->set_max_files_to_load( readme.max_files_to_load() );
    if (readme.data_local_filedir() != "") {
      reader->set_local_file_dir( readme.data_local_filedir() );
    }

    if (create_tarball) {
      if (opts->has_int("test_tarball")) {
        reader->set_absolute_sample_count( opts->get_int("test_tarball"));
        reader->set_use_percent( 0. );
        reader->set_first_n(0);
      } else {
        reader->set_absolute_sample_count( 0. );
        reader->set_use_percent( 1.0 );
        reader->set_first_n( 0 );
      }
    } else {
      reader->set_absolute_sample_count( readme.absolute_sample_count() );
      reader->set_use_percent( readme.percent_of_data_to_use() );
      reader->set_first_n( readme.first_n() );

      reader->set_gan_labelling(readme.gan_labelling());
      reader->set_gan_label_value(readme.gan_label_value());

      reader->set_partitioned(readme.is_partitioned(), readme.partition_overlap(), readme.partition_mode());
    }

    if (readme.role() == "train") {
      reader->set_role("train");
    } else if (readme.role() == "test") {
      reader->set_role("test");
    } else if (readme.role() == "validate") {
      reader->set_role("validate");
    } else {
      reader->set_role("error");
    }
    if (readme.role() == "train") {
      if (create_tarball || separate_validation) {
        reader->set_validation_percent( 0. );
      } else {
        reader->set_validation_percent( readme.validation_percent() );
      }
    }

    reader->set_master(master);

    reader->load();

    if (readme.role() == "train") {
      data_readers[execution_mode::training] = reader;
    } else if (readme.role() == "test") {
      // While the default validation_percent is 0.0, this line is added to be consistent with the case of "train"
      reader->set_validation_percent( 0. );
      data_readers[execution_mode::testing] = reader;
    } else if (readme.role() == "validate") {
      reader->set_validation_percent( 0. );
      data_readers[execution_mode::validation] = reader;
    }

    if (readme.role() == "train" && readme.validation_percent() > 0. && !create_tarball && !separate_validation) {
      if (name == "mnist") {
        reader_validation = new mnist_reader(shuffle);
        (*(mnist_reader *)reader_validation) = (*(mnist_reader *)reader);
      } else if (name == "numpy_npz_conduit_reader") {
        reader_validation = new numpy_npz_conduit_reader(*dynamic_cast<const numpy_npz_conduit_reader*>(reader));
      } else if (name == "imagenet") {
        reader_validation = new imagenet_reader(*dynamic_cast<const imagenet_reader*>(reader));
      } else if (name == "smiles") {
        reader_validation = new smiles_data_reader(*dynamic_cast<const smiles_data_reader*>(reader));
      } else if (name == "jag_conduit") {
        /// If the training data reader was shared and the validate reader is split from it, then the validation data reader
        /// is also shared
        if(is_shareable_training_data_reader) {
          const std::string role = "validate";
          if (!peek_map(leading_reader_jag_conduit, role)) {
            reader_validation = new data_reader_jag_conduit(*dynamic_cast<const data_reader_jag_conduit*>(reader));
            auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader_validation);
            reader_jag_conduit->set_leading_reader(reader_jag_conduit);
            reader_jag_conduit->set_role(role);
            leading_reader_jag_conduit[role] = reader_jag_conduit;
          } else {
            // Copy construct the leading validation reader into another validation reader.
            // We do not copy the train reader as the subset of data may already have been
            // assigned to validation reader when validation percent is set.
            // Thus, we need to avoid taking a subset of a subset.
            const auto leader = peek_map(leading_reader_jag_conduit, role);
            reader_validation = new data_reader_jag_conduit(*leader);
            auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader_validation);
            reader_jag_conduit->set_leading_reader(leader);
          }
        } else {
          reader_validation = new data_reader_jag_conduit(*dynamic_cast<const data_reader_jag_conduit*>(reader));
          const std::string role = "validate";
          auto reader_jag_conduit = dynamic_cast<data_reader_jag_conduit*>(reader_validation);
          reader_jag_conduit->set_leading_reader(reader_jag_conduit);
          reader_jag_conduit->set_role(role);
          leading_reader_jag_conduit[role] = reader_jag_conduit;
        }
      } else if (name == "ras_lipid") {
        auto *ras_lipid = new ras_lipid_conduit_data_reader(shuffle);
        ras_lipid->set_num_labels(readme.num_labels());
        reader_validation = ras_lipid;
        (*(ras_lipid_conduit_data_reader *)reader_validation) = (*(ras_lipid_conduit_data_reader *)reader);
      } else if (name == "nci") {
        reader_validation = new data_reader_nci(shuffle);
        (*(data_reader_nci *)reader_validation) = (*(data_reader_nci *)reader);
      } else if (name == "csv") {
        reader_validation = new csv_reader(shuffle);
        (*(csv_reader *)reader_validation) = (*(csv_reader *)reader);
      } else if (name == "numpy") {
        reader_validation = new numpy_reader(shuffle);
        (*(numpy_reader *)reader_validation) = (*(numpy_reader *)reader);
      } else if (name == "merge_samples") {
        reader_validation = new data_reader_merge_samples(*(data_reader_merge_samples *)reader);
      } else if (name == "merge_features") {
        reader_validation = new data_reader_merge_features(*(data_reader_merge_features *)reader);
      } else if (name == "cifar10") {
        reader_validation = new cifar10_reader(shuffle);
        (*(cifar10_reader *)reader_validation) = (*(cifar10_reader *)reader);
      } else if (name == "synthetic") {
        reader_validation = new data_reader_synthetic(*(data_reader_synthetic *)reader);
        (*(data_reader_synthetic *) reader_validation) = (*(data_reader_synthetic *)reader);
      } else if (name == "mesh") {
        reader_validation = new mesh_reader(shuffle);
        (*(mesh_reader *)reader_validation) = (*(mesh_reader *)reader);
      } else if (name == "python") {
#ifdef LBANN_HAS_PYTHON
        const auto& params = readme.python();
        reader_validation = new python_reader(params.module(),
                                              params.module_dir(),
                                              params.sample_function(),
                                              params.num_samples_function(),
                                              params.sample_dims_function(),
                                              shuffle);
        (*(python_reader *)reader_validation) = (*(python_reader *)reader);
#else
        LBANN_ERROR("attempted to construct Python data reader, "
                    "but LBANN is not built with Python/C API");
#endif // LBANN_HAS_PYTHON
      }

      reader_validation->set_role("validate");
      reader_validation->use_unused_index_set();
      data_store_conduit *store = reader_validation->get_data_store_ptr();
      if (store != nullptr) {
        store->set_data_reader_ptr(reader_validation);
        reader_validation->get_data_store_ptr()->compact_nodes();
      }

      size_t ntrain = reader->get_num_data();
      if (ntrain == 0) {
        LBANN_ERROR("num train samples = 0; something is wrong");
      }

      if (master) {
        size_t num_train = reader->get_num_data();
        size_t num_validate = reader_validation->get_num_data();
        double validate_percent = ((double) num_validate / (double) (num_train+num_validate))*100.0;
        double train_percent = ((double) num_train / (double) (num_train+num_validate))*100.0;
        std::cout << "Training using " << train_percent << "% of the training data set, which is " << reader->get_num_data() << " samples." << std::endl
                  << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->get_num_data() << " samples.";
        if (name == "jag_conduit") {
          std::cout << " jag conduit leading reader " << dynamic_cast<data_reader_jag_conduit*>(reader)->get_leading_reader()
                    << " of " << (is_shareable_training_data_reader? "shared" : "unshared") << " reader " << reader << " for " << reader->get_role() << std::endl;
        }
        std::cout << std::endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }

  if (master) {
    if (separate_validation) {
      const generic_data_reader* r_train = peek_map(data_readers, execution_mode::training);
      const generic_data_reader* r_validate = peek_map(data_readers, execution_mode::validation);
      const size_t num_train = (r_train == nullptr)? 0u : r_train->get_num_data();
      const size_t num_validate = (r_validate == nullptr)? 0u : r_validate->get_num_data();
      std::cout << "Training using " << num_train << " samples." << std::endl
                << "Validating using " << num_validate << " samples." << std::endl;
    }
    const generic_data_reader* r_test = peek_map(data_readers, execution_mode::testing);
    const size_t num_test = (r_test == nullptr)? 0u : r_test->get_num_data();
    std::cout << "Testing using " << num_test << " samples." << std::endl;
  }
  // remove null data_reader pointers if there is any
  for (auto it = data_readers.cbegin(); it != data_readers.cend() ; ) {
    if (!it->second) {
      it = data_readers.erase(it);
    } else {
      ++it;
    }
  }
}

void read_prototext_file(const std::string& fn, lbann_data::LbannPB& pb, const bool master)
{
  std::ostringstream err;
  int fd = open(fn.c_str(), O_RDONLY);
  if (fd == -1) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
      throw lbann_exception(err.str());
    }
  }
  using FIS=google::protobuf::io::FileInputStream;
  auto input = std::unique_ptr<FIS, std::function<void(FIS*)>>(
    new google::protobuf::io::FileInputStream(fd),
    [](FIS* x) {
      x->Close();
      delete x;
    });
  bool success = google::protobuf::TextFormat::Parse(input.get(), &pb);
  if (!success) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << std::endl;
      throw lbann_exception(err.str());
    }
  }
}

bool write_prototext_file(const std::string& fn, lbann_data::LbannPB& pb)
{
  int fd = open(fn.c_str(), O_APPEND | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    return false;
  }
  auto *output = new google::protobuf::io::FileOutputStream(fd);
  if (!google::protobuf::TextFormat::Print(pb, output)) {
    close(fd);
    delete output;
    return false;
  }
  delete output;
  close(fd);
  return true;
}

bool check_if_num_parallel_readers_set(const lbann_comm& comm, const lbann_data::Trainer& trainer)
{
  const bool master = comm.am_world_master();
  const int parallel_io = trainer.num_parallel_readers();

  if (parallel_io == 0) {
    if (master) {
      std::cout << "\tMax Parallel I/O Fetch: " << comm.get_procs_per_trainer() <<
        " (Limited to # Processes)" << std::endl;
    }
    return false;
  }
  if (master) {
    std::cout << "\tMax Parallel I/O Fetch: " << parallel_io << std::endl;
  }
  return true;
}

void set_num_parallel_readers(const lbann_comm& comm, lbann_data::LbannPB& p)
{
  lbann_data::Trainer *trainer = p.mutable_trainer();
  const bool is_set = check_if_num_parallel_readers_set(comm, *trainer);

  if (!is_set) {
    const int parallel_io = comm.get_procs_per_trainer();
    trainer->set_num_parallel_readers(parallel_io); //adjust the prototext
  }
}

int get_requested_num_parallel_readers(const lbann_comm& comm, const lbann_data::LbannPB& p)
{
  const lbann_data::Trainer& trainer = p.trainer();
  const bool is_set = check_if_num_parallel_readers_set(comm, trainer);

  if (!is_set) {
    return comm.get_procs_per_trainer();
  }
  return trainer.num_parallel_readers();
}

void set_data_readers_filenames(
  const std::string& which, lbann_data::LbannPB& p)
{
  options *opts = options::get();
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    if (r->role() == which) {
      std::ostringstream s;
      s << "data_filedir_" << which;
      if (opts->has_string(s.str())) {
        r->set_data_filedir(opts->get_string(s.str()));
      }else {
        s.clear();
        s.str("");
        s << "data_filedir";
        if (opts->has_string(s.str())) {
          r->set_data_filedir(opts->get_string(s.str()));
        }
      }
      s.clear();
      s.str("");
      s << "data_filename_" << which;
      if (opts->has_string(s.str())) {
        r->set_data_filename(opts->get_string(s.str()));
      }
      s.clear();
      s.str("");
      s << "label_filename_" << which;
      if (opts->has_string(s.str())) {
        r->set_label_filename(opts->get_string(s.str()));
      }
    }
  }
}

void set_data_readers_index_list(
  const std::string& which, lbann_data::LbannPB& p)
{
  options *opts = options::get();
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  const std::string key_role = "index_list_" + which;

  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    if (r->role() == which) {
      r->set_index_list(opts->get_string(key_role));
    }
  }
}

void set_data_readers_percent(lbann_data::LbannPB& p)
{
  options *opts = options::get();
  double percent = opts->get_float("data_reader_percent");
  if (percent <= 0 || percent > 1.0) {
      std::ostringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << " --data_reader_percent=<float> must be > 0 and <= 1.0";
      throw lbann_exception(err.str());
  }
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    r->set_percent_of_data_to_use( percent );
  }
}

void customize_data_readers_index_list(const lbann_comm& comm, lbann_data::LbannPB& p)
{
  lbann_data::DataReader *readers = p.mutable_data_reader();
  const lbann_data::Model& pb_model = p.model();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    std::ostringstream s;
    std::string basename = get_basename_without_ext(r->index_list());
    std::string ext = get_ext_name(r->index_list());
    if(r->index_list_per_model()) {
      s << pb_model.name() << "_";
    }
    if(r->index_list_per_trainer()) {
      s << "t" << comm.get_trainer_rank() << "_";
    }
    s << basename;
    s << "." << ext;
    r->set_index_list(s.str());
  }
}

void get_cmdline_overrides(const lbann_comm& comm, lbann_data::LbannPB& p)
{
  std::ostringstream err;

  options *opts = options::get();
  lbann_data::Trainer *trainer = p.mutable_trainer();
  lbann_data::Model *model = p.mutable_model();
  lbann_data::DataReader *d_reader = p.mutable_data_reader();
  int size = d_reader->reader_size();

  if (opts->has_int("absolute_sample_count")) {
    for (int j=0; j<size; j++) {
      int n = opts->get_int("absolute_sample_count");
      lbann_data::Reader *readme = d_reader->mutable_reader(j);
      readme->set_percent_of_data_to_use(0.0);
      readme->set_absolute_sample_count(n);
    }
  }

  if (opts->has_string("data_filedir")
      or opts->has_string("data_filedir_train")
      or opts->has_string("data_filename_train")
      or opts->has_string("label_filename_train")) {
    set_data_readers_filenames("train", p);
  }
  if (opts->has_string("data_filedir")
      or opts->has_string("data_filedir_test")
      or opts->has_string("data_filename_test")
      or opts->has_string("label_filename_test")) {
    set_data_readers_filenames("test", p);
  }
  if (opts->has_string("index_list_train")) {
    set_data_readers_index_list("train", p);
  }
  if (opts->has_string("index_list_test")) {
    set_data_readers_index_list("test", p);
  }
  if (opts->has_string("data_reader_percent")) {
    set_data_readers_percent(p);
  }
  if (opts->get_bool("no_im_comm")) {
    int sz = model->callback_size();
    for (int j=0; j<sz; j++) {
      lbann_data::Callback *c = model->mutable_callback(j);
      if (c->has_imcomm()) {
        c->clear_imcomm();
      }
    }
  }
  if (opts->has_int("mini_batch_size")) {
    trainer->set_mini_batch_size(opts->get_int("mini_batch_size"));
  }
  if (opts->has_int("num_epochs")) {
    model->set_num_epochs(opts->get_int("num_epochs"));
  }
  if (opts->has_int("hydrogen_block_size")) {
    trainer->set_hydrogen_block_size(opts->get_int("hydrogen_block_size"));
  }
  if (opts->has_int("procs_per_trainer")) {
    trainer->set_procs_per_trainer(opts->get_int("procs_per_trainer"));
  }
  if (opts->has_int("num_parallel_readers")) {
    trainer->set_num_parallel_readers(opts->get_int("num_parallel_readers"));
  }
  if (opts->get_bool("disable_cuda")) {
    model->set_disable_cuda(opts->get_bool("disable_cuda"));
  }
  if (opts->has_int("random_seed")) {
    trainer->set_random_seed(opts->get_int("random_seed"));
  }
  if(opts->get_bool("serialize_io")) {
    model->set_serialize_io(opts->get_bool("serialize_io"));
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

  const lbann_data::Trainer &t = p.trainer();
  const lbann_data::Model &m = p.model();

  bool disable_cuda = m.disable_cuda();
#ifndef LBANN_HAS_GPU
  disable_cuda = true;
#endif // LBANN_HAS_GPU
  bool disable_cudnn = disable_cuda;
#ifndef LBANN_HAS_CUDNN
  disable_cudnn = true;
#endif // LBANN_HAS_CUDNN
  bool enable_determinism = false;
#ifdef LBANN_DETERMINISTIC
  enable_determinism = true;
#endif // LBANN_DETERMINISTIC

  std::cout << std::endl
            << "Running with these parameters:\n"
            << " General:\n"
            << "  datatype size:              " << sizeof(DataType) << std::endl
            << "  mini_batch_size:            " << t.mini_batch_size() << std::endl
            << "  num_epochs:                 " << m.num_epochs()  << std::endl
            << "  hydrogen_block_size:        " << t.hydrogen_block_size()  << std::endl
            << "  procs_per_trainer:          " << t.procs_per_trainer()  << std::endl
            << "  num_parallel_readers:       " << t.num_parallel_readers()  << std::endl
            << "  serialize_io:               " << m.serialize_io()  << std::endl
            << "  cuda:                       " << (disable_cuda ? "disabled" : "enabled") << std::endl
            << "  cudnn:                      " << (disable_cudnn ? "disabled" : "enabled") << std::endl;
  auto& arg_parser = global_argument_parser();
  std::stringstream root_rng, rng, data_seq_rng;
  for(size_t i = 0; i < random_seeds.size(); i++) {
    int trainer_rank = comm.map_world_rank_to_trainer_rank(i);
    int rank_in_trainer = comm.map_world_rank_to_rank_in_trainer(i);
    if(rank_in_trainer < arg_parser.get<int>(MAX_RNG_SEEDS_DISPLAY)) {
      std::stringstream id;
      id << "[" << trainer_rank << "][" << rank_in_trainer << "]";
      root_rng << id.str() << "=" << std::setfill('0') << std::setw(10) << static_cast<unsigned int>(root_random_seeds[i]) << " " ;
      rng << id.str() << "=" << std::setfill('0') << std::setw(10) << static_cast<unsigned int>(random_seeds[i]) << " " ;
      data_seq_rng << id.str() << "=" << std::setfill('0') << std::setw(10) << static_cast<unsigned int>(data_seq_random_seeds[i]) << " " ;
    }else {
      root_rng << "... ";
      rng << "... ";
      data_seq_rng << "... ";
    }
  }
  std::cout << "  root_random_seed[t][r]:     " << root_rng.str() << std::endl;
  std::cout << "  random_seed[t][r]:          " << rng.str() << std::endl;
  std::cout << "  data_seq_random_seed[t][r]: " << data_seq_rng.str() << std::endl;
  std::cout << "  deterministic_exec:         " << (enable_determinism ? "enabled" : "disabled") << std::endl
            << "  data_layout:                " << m.data_layout()  << std::endl
            << "     (only used for metrics)\n";
}

void print_help(const lbann_comm& comm)
{
  if (comm.am_world_master()) {
    print_help(std::cerr);
  }
}

void print_help(std::ostream& os)
{
  os <<
       "General usage: you need to specify three prototext files, e.g:\n"
       "  srun -n# proto --model=<string> --optimizer=<string> --reader=<string> --metadata=<string>\n"
       "\n"
       "  However, if you are re-running an experiment from a previously saved\n"
       "  file, you only need to specify --prototext=<string>\n"
       "  When proto is run, an output file containing the concatenated prototext\n"
       "  files, along with other data is written. The default name for this file\n"
       "  is 'data.prototext'  You can specify an alternative name via the option:\n"
       "  --saveme=<string>  You can suppress writing the file via the option:\n"
       "  --saveme=0\n"
       "\n"
       "Some prototext values can be overriden on the command line;\n"
       "(notes: use '1' or '0' for bool; if no value is given for a flag,\n"
       "        e.g: --disable_cuda, then a value of '1' is assigned)\n"
       "\n"
       "General:\n"
       "  --mini_batch_size=<int>\n"
       "  --num_epochs=<int>\n"
       "  --hydrogen_block_size=<int>\n"
       "  --procs_per_trainer=<int>\n"
       "  --num_parallel_readers=<int>\n"
       "  --serialize_io=<bool>\n"
       "      force data readers to use a single thread for I/O\n"
       "  --disable_background_io_activity=<bool>\n"
       "      prevent the input layers from fetching data in the background\n"
       "  --disable_cuda=<bool>\n"
       "     has no effect unless lbann was compiled with: LBANN_HAS_CUDNN\n"
       "  --random_seed=<int>\n"
       "  --objective_function<string>\n"
       "      <string> must be: categorical_cross_entropy or mean_squared_error\n"
       "  --data_layout<string>\n"
       "      <string> must be: data_parallel or model_parallel\n"
       "      note: this will be applied to all layers, metrics (and others)\n"
       "            that take DATA_PARALLEL or MODEL_PARALLEL as a template parameter\n"
       "  --print_affinity\n"
       "      display information on how OpenMP threads are provisioned\n"
       "  --use_data_store \n"
       "      Enables the data store in-memory structure\n"
       "  --preload_data_store \n"
       "      Preloads the data store in-memory structure during data reader load time\n"
       "  --super_node \n"
       "      Enables the data store in-memory structure to use the supernode exchange structure\n"
       "  --write_sample_list \n"
       "      Writes out the sample list that was loaded into the current directory\n"
       "  --ltfb_verbose \n"
       "      Increases number of per-trainer messages that are reported\n"
       "  --ckpt_dir=<string>\n"
       "      Save to or restart from a specific checkpoint directory.\n"
       "      Additionally, sets the output directory for dumping weights.\n"
       "      Modifies callbacks: checkpoint, save_model, dump_weights\n"
       "  --restart_dir=<string>\n"
       "      Restart from a checkpoint found in the given directory.\n"
       "      If the directory doesn't exist or doesn't contain a checkpoint,\n"
       "      an error will be thrown.\n"
       "  --load_model_weights_dir=<string>\n"
       "      Load model wieghts found in the given directory.\n"
       "      If the directory doesn't exist, doesn't contain valid weights,\n"
       "      or doesn't contain a checkpoint,\n"
       "      an error will be thrown.\n"
       "  --load_model_weights_dir_is_complete=<bool>\n"
       "      Use load_model_weights_dir as given, ignoring checkpoint hierarchy.\n"
       "\n"
       "DataReaders:\n"
       "  --data_filedir=<string>\n"
       "      sets the file directory for train and test data\n"
       "  --data_filedir_train=<string>   --data_filedir_test=<string>\n"
       "  --data_filename_train=<string>  --data_filename_test=<string>\n"
       "  --index_list_train=<string>     --index_list_test=<string>\n"
       "  --label_filename_train=<string> --label_filename_test=<string>\n"
       "  --data_reader_percent=<float>\n"
       "  --share_testing_data_readers=<bool:[0|1]>\n"
       "\n"
       "Callbacks:\n"
       "  --image_dir=<string>\n"
       "      if the model has callback_save_images, this determines where the\n"
       "      images are saved\n"
       "  --no_im_comm=<bool>\n"
       "      removes ImComm callback, if present; this is intended for\n"
       "      running alexnet with a single model, but may be useful elsewhere\n"
       "\n"
       "Optimizers; all values except for nesterov are floats;\n"
       "            the values shown in <...> are the default values, that will be\n"
       "            used if the option is not specified on the cmd line.\n"
       "            If you specify an option that is not applicable to your choice\n"
       "            of optimizer, the option is ignored\n"
       "\n";
}

void copy_file(std::string fn, std::ofstream &out)
{
  std::ifstream in(fn.c_str());
  if (!in.is_open()) {
    std::ostringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for reading: " << fn;
    throw std::runtime_error(err.str());
  }
  std::ostringstream s;
  s << in.rdbuf();
  out << s.str();
}

void save_session(const lbann_comm& comm, const int argc, char * const* argv, lbann_data::LbannPB& p)
{
  if (!comm.am_world_master()) {
    return;
  }

  options *opts = options::get();

  //do not write output file for a repeated experiment;
  //may want to revisit this decision later ...
  if (opts->has_string("prototext")) {
    return;
  }

  //setup file name
  // Note: If the file name is not unique, append numbers until it is.
  std::string model_name = p.model().name();
  if (model_name.empty()) { model_name = "model"; };
  std::string file_name = model_name + ".prototext";
  El::Int file_name_index = 1;
  while (std::ifstream(file_name.c_str())) {
    file_name_index++;
    file_name = (model_name
                 + "_" + std::to_string(file_name_index)
                 + ".prototext");
  }

  //open output file
  std::ofstream out(file_name.c_str());
  if (!out.is_open()) {
    std::ostringstream err;
    err << "failed to open file (" << file_name << ") for writing";
    LBANN_ERROR(err.str());
  }
  std::cout << std::endl << "writing options and prototext to file: " << file_name << "\n\n";

  //output all data
  out << "# cmd line for original experiment:\n#  $ ";
  for (int h=0; h<argc; h++) {
    out << argv[h] << " ";
  }
  std::string lbann_version("unknown: LBANN_VERSION is not defined");

#ifdef LBANN_VERSION
  lbann_version = LBANN_MAKE_STR(LBANN_VERSION);
#endif

  std::time_t r = std::time(nullptr);
  char *tm = std::ctime(&r);
  size_t fixme = strlen(tm);
  tm[fixme-1] = 0;
  out << "\n#\n# Experiment conducted at: "
      <<  tm
      << "\n#\n#\n# Experiment was run with lbann version: "
      << lbann_version << "\n#\n#\n# To rerun the experiment: \n"
      << "#  $ srun -n" << comm.get_procs_in_world() << " " << argv[0]
      << " --prototext=" << file_name << "\n#\n#\n";

  out << "# Selected SLURM Environment Variables:\n";
  std::vector<std::string> v = {"HOST", "SLURM_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE"};
  for (auto & i : v) {
    char *c = std::getenv(i.c_str());
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
  if (str.size() == 0) return std::string();

  const std::string whitespace = "\f\n\r\t\v ";
  auto first = str.find_first_not_of(whitespace);

  // All characters are whitespace; short-circuit.
  if (first == std::string::npos) return std::string();

  auto last = str.find_last_not_of(whitespace);
  return str.substr(first, (last-first)+1);
}

} // namespace lbann
