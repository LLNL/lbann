#include "lbann/proto/proto_common.hpp"

#include "lbann/lbann.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/proto/init_image_data_readers.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

using namespace lbann;

/** Map from layer names to layers. */
std::map<std::string, Layer*> model_layers;
/** List of layer names. */
std::vector<std::string> model_layer_names;

/** Map from weights names to weights. */
std::map<std::string, weights*> model_weights;
/** List of weights names. */
std::vector<std::string> model_weights_names;

/** Whether a layer is already in the model. */
inline bool layer_is_in_model(std::string name) {
  return model_layers.find(name) != model_layers.end();
}

/** Whether a set of weights are already in the model. */
inline bool weights_are_in_model(std::string name) {
  return model_weights.find(name) != model_weights.end();
}

bool has_motifs(lbann_comm *comm, const lbann_data::LbannPB& p) {
  bool master = comm->am_world_master();
  if (master) std::cout << "starting has_motifs\n";
  const lbann_data::Model& m = p.model();
  const int num_layers = m.layer_size();
  for (int j=0; j<num_layers; j++) {
    const lbann_data::Layer& layer = m.layer(j);
    if (layer.has_motif_layer()) {
      return true;
    }
  }
  return false;
}

void expand_motifs(lbann_comm *comm, lbann_data::LbannPB& pb) {
  bool master = comm->am_world_master();
  if (master) std::cout << "starting expand_motifs\n";
  const lbann_data::MotifDefinitions& m = pb.motif_definitions();
  const int num_motifs = m.motif_size();
  for (int j=0; j<num_motifs; j++) {
  }
}


void setup_pointers(
  std::vector<lbann_data::Layer> &proto_layers,
  lbann::model *model,
  bool master)
{
  for (size_t i=0; i<proto_layers.size(); i++) {
    std::string layer_name = model_layer_names[i];
    Layer *layer = model_layers[layer_name];

    std::stringstream err;

    // Set layer parents
    {
      std::string name;
      std::stringstream ss(proto_layers[i].parents());
      while (ss >> name) {
        if (master and not layer_is_in_model(name)) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << "could not find parent layer " << name;
          throw lbann_exception(err.str());
        }
        Layer *parent_layer = model_layers[name];
        layer->add_parent_layer(parent_layer);
      }
      if (ss.bad()) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not parse " << proto_layers[i].parents();
        throw lbann_exception(err.str());
      }
    }

    // Set layer children
    {
      std::string name;
      std::stringstream ss(proto_layers[i].children());
      while (ss >> name) {
        if (master and not layer_is_in_model(name)) {
          err << __FILE__ << " " << __LINE__ << " :: "
              << "could not find child layer " << name;
          throw lbann_exception(err.str());
        }
        Layer *child_layer = model_layers[name];
        layer->add_child_layer(child_layer);
      }
      if (ss.bad()) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not parse " << proto_layers[i].children();
        throw lbann_exception(err.str());
      }
    }

    // Set a target layer's paired input layer
    if (dynamic_cast<target_layer*>(layer) != nullptr) {
      auto *target = dynamic_cast<target_layer*>(layer);

      std::string name;

      // Get input layer name
      if (proto_layers[i].has_target_distributed_minibatch()) {
        name = proto_layers[i].target_distributed_minibatch().paired_input_layer();
      }
      if (proto_layers[i].has_target_partitioned_minibatch()) {
        name = proto_layers[i].target_partitioned_minibatch().paired_input_layer();
      }
      if (name.empty()) {
        for (auto& other_layer : model_layers) {
          if (dynamic_cast<input_layer*>(other_layer.second) != nullptr) {
            name = other_layer.first;
            break;
          }
        }
      }
      if (master and (name.empty() or not layer_is_in_model(name))) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not find paired input layer for target layer";
        throw lbann_exception(err.str());
      }

      // Set input layer
      auto *input = dynamic_cast<input_layer*>(model_layers[name]);
      target->set_paired_input_layer(input);

    }

    // Set a reconstruction layer's original layer
    if (proto_layers[i].has_reconstruction()) {

      std::string name;

      // Get original layer name
      name = proto_layers[i].reconstruction().original_layer();
      if (name.empty()) {
        for (auto& other_layer : model_layers) {
          if (dynamic_cast<input_layer*>(other_layer.second) != nullptr) {
            name = other_layer.first;
            break;
          }
        }
      }
      if (master and (name.empty() or not layer_is_in_model(name))) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not find original layer for reconstruction layer";
        throw lbann_exception(err.str());
      }

#if 0
      // Set original layer
      Layer *original_layer = model_layers[name];
      if (dynamic_cast<reconstruction_layer<data_layout::MODEL_PARALLEL>*>(layer)) {
        auto *reconstruction
          = dynamic_cast<reconstruction_layer<data_layout::MODEL_PARALLEL>*>(layer);
        reconstruction->set_original_layer(original_layer);
      }
      if (dynamic_cast<reconstruction_layer<data_layout::DATA_PARALLEL>*>(layer)) {
        auto *reconstruction
          = dynamic_cast<reconstruction_layer<data_layout::DATA_PARALLEL>*>(layer);
        reconstruction->set_original_layer(original_layer);
      }
#endif

    }

  }
}

#if 0
lbann_callback_imcomm::comm_type get_comm_type(const std::string &s, bool master)
{
  if (s == "none") {
    return lbann_callback_imcomm::comm_type::NONE;
  } else if (s == "normal") {
    return lbann_callback_imcomm::comm_type::NORMAL;
  } else if (s == "onebit_quantization") {
    return lbann_callback_imcomm::comm_type::ONEBIT_QUANTIZATION;
  } else if (s == "thresh_quantization") {
    return lbann_callback_imcomm::comm_type::THRESH_QUANTIZATION;
  } else if (s == "adaptive_quantization") {
    return lbann_callback_imcomm::comm_type::ADAPTIVE_QUANTIZATION;
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " <<__LINE__
         << " :: unkown comm_type: " << s
          << " should be one of: none, normal, onebit_quantization, thresh_quantization, adaptive_quantization";
      throw lbann_exception(err.str());
    } else {
      return lbann_callback_imcomm::comm_type::NONE; //keep compiler happy, and have only one proc throw exception
    }
  }
}
#endif

pool_mode get_pool_mode(const std::string& s, bool master)
{
  if (s == "max") {
    return pool_mode::max;
  } else if (s == "average") {
    return pool_mode::average;
  } else if (s == "average_no_pad") {
    return pool_mode::average_no_pad;
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " <<__LINE__
          << " :: unkown pool_mode: " << s
          << " should be one of: max, average, average_no_pad";
      throw lbann_exception(err.str());
    } else {
    return pool_mode::max; //keep compiler happy, and have only one proc throw exception
    }
  }
}

inline data_layout get_data_layout(const std::string& s)
{
  if (s == "model_parallel") {
    return data_layout::MODEL_PARALLEL;
  } else if (s == "data_parallel") {
    return data_layout::DATA_PARALLEL;
  } else {
    return data_layout::invalid;
  }
}


void get_proto_layers(
  std::vector<lbann_data::Layer> &proto_layers,
  const lbann_data::Model& m,
  lbann_comm *comm)
{
  const bool master = comm->am_world_master();
  const int num_layers = m.layer_size();
  std::stringstream err;
  proto_layers.clear();

  for (int j=0; j<num_layers; j++) {
    const lbann_data::Layer& layer = m.layer(j);

    //ensure no whitespace in name
    std::stringstream s(layer.name());
    std::string token;
    int num_tokens = 0;
    while (s >> token) {
      ++num_tokens;
    }
    if (master and num_tokens > 1) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << " layer name \"" << layer.name() << "\" is invalid. "
          << "Cannot contain whitespace.";
      throw lbann_exception(err.str());
    }

    //add layer to list of layers
    proto_layers.push_back(layer);
  }
}


void add_layers(
  lbann::model *model,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  cudnn::cudnn_manager *cudnn,
  const lbann_data::LbannPB& p)
{
  lbann_comm *comm = model->get_comm();
  const bool master = comm->am_world_master();
  if (master) {
    std::cout << "starting add_layers\n";
  }

  std::stringstream err;
  model_layers.clear(); //shouldn't need this, but just in case ...
  model_layer_names.clear();

  const lbann_data::Model& m = p.model();
  std::vector<lbann_data::Layer> proto_layers;
  get_proto_layers(proto_layers, m, comm);

  for (lbann_data::Layer& layer : proto_layers) {
    Layer *d = nullptr;

    // Get layer layout
    const data_layout layout = get_data_layout(layer.data_layout());
    if (master and layout == data_layout::invalid) {
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << layer.name() << " "
          << "has invalid data layout " << layer.data_layout();
      throw lbann_exception(err.str());
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: Relu
    //////////////////////////////////////////////////////////////////
    if (layer.has_relu()) {
      //const lbann_data::Relu &ell = layer.relu();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new relu_layer<data_layout::MODEL_PARALLEL>(comm, nullptr);
      } else {
        d = new relu_layer<data_layout::DATA_PARALLEL>(comm, cudnn);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: sigmoid
    //////////////////////////////////////////////////////////////////
    else if (layer.has_sigmoid()) {
      //const lbann_data::Sigmoid &ell = layer.sigmoid();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new sigmoid_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new sigmoid_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }
#if 0
    //////////////////////////////////////////////////////////////////
    // LAYER: reconstruction
    //////////////////////////////////////////////////////////////////
    else if (layer.has_reconstruction()) {
      //xxx const lbann_data::TargetReconstruction & ell = layer.reconstruction();
      /*
      std::string original_layer = ell.original_layer();
      if (the_layers.find(original_layer) == the_layers.end() and master) {
        err << __FILE__ << " " << __LINE__ << " :: the original_field in the "
            << " Reconstruction layer has index " << original_layer
            << " but we don't have a layer with that index. Something may be "
            << " wrong in your prototext file";
        throw lbann_exception(err.str());
      }
      */
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new reconstruction_layer<data_layout::MODEL_PARALLEL>(
          comm,
          nullptr
        );
      } else {
        d = new reconstruction_layer<data_layout::DATA_PARALLEL>(
          comm,
          nullptr
        );
      }
    }
#endif
    //////////////////////////////////////////////////////////////////
    // LAYER: input_distributed_minibatch
    //////////////////////////////////////////////////////////////////
    else if (layer.has_input_distributed_minibatch()) {
      const lbann_data::InputDistributedMiniBatch& ell = layer.input_distributed_minibatch();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers,
          !ell.data_set_per_model());
      } else {
        d = new input_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers,
          !ell.data_set_per_model());
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: input_partitioned_minibatch
    //////////////////////////////////////////////////////////////////
    else if (layer.has_input_partitioned_minibatch()) {
      const lbann_data::InputPartitionedMiniBatch& ell = layer.input_partitioned_minibatch();
      if (layout == data_layout::MODEL_PARALLEL and master) {
        err << __FILE__ << " " << __LINE__ << " :: input_layer_partitioned_minibatch "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new input_layer_partitioned_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers,
          !ell.data_set_per_model());
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: fully_connected
    //////////////////////////////////////////////////////////////////
    else if (layer.has_fully_connected()) {
      const lbann_data::FullyConnected& ell = layer.fully_connected();
      int num_neurons;
      if (layer.num_neurons_from_data_reader()) {
        num_neurons = data_readers[execution_mode::training]->get_linearized_data_size();
      } else {
        num_neurons = ell.num_neurons();
      }
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
          comm,
          num_neurons,
          nullptr,
          ell.has_bias(),
          cudnn);
      } else {
        d = new fully_connected_layer<data_layout::DATA_PARALLEL>(
          comm,
          num_neurons,
          nullptr,
          ell.has_bias(),
          cudnn);
      }

    }

    //////////////////////////////////////////////////////////////////
    // LAYER: reshape
    //////////////////////////////////////////////////////////////////
    else if (layer.has_reshape()) {
      const lbann_data::Reshape &ell = layer.reshape();
      int i, num_dims;
      std::vector<int> dims;
      if (layer.num_neurons_from_data_reader()) {
        if(ell.reshape_to_flattened_conv_format()) {
          dims.push_back(1);
          num_dims = 2;
        }else {
          num_dims = 1;
        }
        dims.push_back(data_readers[execution_mode::training]->get_linearized_data_size());

        if(ell.num_dims() != 0 || ell.dims() != "") {
          err << __FILE__ << " " << __LINE__ << " :: reshape illegal combination using"
              << "num_neurons_from_data_reader flag with num_dims or dims fields";
          throw lbann_exception(err.str());
        }
      } else {
        std::stringstream s(ell.dims());
        while (s >> i) {
          dims.push_back(i);
        }
        num_dims = ell.num_dims();
      }
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new reshape_layer<data_layout::MODEL_PARALLEL>(comm, num_dims, dims.data());
      } else {
        d = new reshape_layer<data_layout::DATA_PARALLEL>(comm, num_dims, dims.data());
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: concatenation
    //////////////////////////////////////////////////////////////////
    else if (layer.has_concatenation()) {
      const lbann_data::Concatenation &ell = layer.concatenation();
      d = new concatenation_layer<>(comm, ell.concatenation_axis(), cudnn);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: slice
    //////////////////////////////////////////////////////////////////
    else if (layer.has_slice()) {
      const lbann_data::Slice &ell = layer.slice();
      std::stringstream s(ell.slice_points());
      std::vector<int> slice_points;
      int i;
      while (s >> i) {
        slice_points.push_back(i);
      }
      d = new slice_layer<>(comm, ell.slice_axis(), slice_points, cudnn);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: sum
    //////////////////////////////////////////////////////////////////
    else if (layer.has_sum()) {
      d = new sum_layer<>(comm, cudnn);
    }
#if 0
    //////////////////////////////////////////////////////////////////
    // LAYER: noise
    //////////////////////////////////////////////////////////////////
    else if (layer.has_noise()) {
      const lbann_data::Noise& ell = layer.noise();
      d = new noise_layer<>(comm,ell.noise_factor(), cudnn);
    }
#endif
    //////////////////////////////////////////////////////////////////
    // LAYER: split
    //////////////////////////////////////////////////////////////////
    else if (layer.has_split()) {
      d = new split_layer<>(comm, cudnn);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: pooling
    //////////////////////////////////////////////////////////////////
    else if (layer.has_pooling()) {
      const lbann_data::Pooling& ell = layer.pooling();
      bool has_vectors = ell.has_vectors();

      if (has_vectors) {

        int i;
        std::stringstream ss(ell.pool_dims());
        std::vector<int> pool_dims;
        while (ss >> i) {
          pool_dims.push_back(i);
        }

        std::vector<int> pool_pads;
        ss.clear();
        ss.str(ell.pool_pads());
        while (ss >> i) {
          pool_pads.push_back(i);
        }

        std::vector<int> pool_strides;
        ss.clear();
        ss.str(ell.pool_strides());
        while (ss >> i) {
          pool_strides.push_back(i);
        }
        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: pooling_layer "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new pooling_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            pool_dims,
            pool_pads,
            pool_strides,
            get_pool_mode(ell.pool_mode(), master),
            cudnn
          );
        }
      } else {
        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: pooling_layer "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new pooling_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            ell.pool_dims_i(),
            ell.pool_pads_i(),
            ell.pool_strides_i(),
            get_pool_mode(ell.pool_mode(), master),
            cudnn
          );
        }
      }
    }
#if 0
    //////////////////////////////////////////////////////////////////
    // LAYER: unpooling
    //////////////////////////////////////////////////////////////////
    else if (layer.has_unpooling()) {
      const lbann_data::Unpooling& ell = layer.unpooling();
      auto *pl = (pooling_layer<data_layout::DATA_PARALLEL>*)model_layers[ell.pooling_layer()];
      if (layout == data_layout::MODEL_PARALLEL and master) {
        err << __FILE__ << " " << __LINE__ << " :: local_response_normalization "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new unpooling_layer<data_layout::DATA_PARALLEL>(
          comm,
          pl
        );
      }
    }
#endif
    //////////////////////////////////////////////////////////////////
    // LAYER: Convolution
    //////////////////////////////////////////////////////////////////
    else if (layer.has_convolution()) {
      const lbann_data::Convolution& ell = layer.convolution();
      bool has_vectors = ell.has_vectors();

      if (has_vectors) {
        std::vector<int> conv_dims;
        std::stringstream ss;
        int i;
        ss.str(ell.conv_dims());
        while (ss >> i) {
          conv_dims.push_back(i);
        }

        std::vector<int> conv_pads;
        ss.clear();
        ss.str(ell.conv_pads());
        while (ss >> i) {
          conv_pads.push_back(i);
        }

        std::vector<int> conv_strides;
        ss.clear();
        ss.str(ell.conv_strides());
        while (ss >> i) {
          conv_strides.push_back(i);
        }

        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: convolution "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new convolution_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            ell.num_output_channels(),
            conv_dims,
            conv_pads,
            conv_strides,
            ell.has_bias(),
            cudnn
          );
        }
      }

      else {
        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: convolution "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new convolution_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            ell.num_output_channels(),
            ell.conv_dims_i(),
            ell.conv_pads_i(),
            ell.conv_strides_i(),
            ell.has_bias(),
            cudnn
          );
        }
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: Deconvolution
    //////////////////////////////////////////////////////////////////
    else if (layer.has_deconvolution()) {
      const lbann_data::Deconvolution& ell = layer.deconvolution();
      bool has_vectors = ell.has_vectors();

      if (has_vectors) {
        std::vector<int> conv_dims;
        std::stringstream ss;
        int i;
        ss.str(ell.conv_dims());
        while (ss >> i) {
          conv_dims.push_back(i);
        }

        std::vector<int> conv_pads;
        ss.clear();
        ss.str(ell.conv_pads());
        while (ss >> i) {
          conv_pads.push_back(i);
        }

        std::vector<int> conv_strides;
        ss.clear();
        ss.str(ell.conv_strides());
        while (ss >> i) {
          conv_strides.push_back(i);
        }

        int num_neurons;
        if (layer.num_neurons_from_data_reader()) {
          num_neurons = data_readers[execution_mode::training]->get_linearized_data_size();
        } else {
          num_neurons = ell.num_output_channels();
        }

        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: deconvolution "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new deconvolution_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            num_neurons/*ell.num_output_channels()*/,
            conv_dims,
            conv_pads,
            conv_strides,
            ell.has_bias(),
            cudnn
          );
        }
      }

      else {
        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: deconvolution "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new deconvolution_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            ell.num_output_channels(),
            ell.conv_dims_i(),
            ell.conv_pads_i(),
            ell.conv_strides_i(),
            ell.has_bias(),
            cudnn
          );
        }
      }
    }
    //////////////////////////////////////////////////////////////////
    // LAYER: local_response_normalization
    //////////////////////////////////////////////////////////////////
    else if (layer.has_local_response_normalization()) {
      const lbann_data::LocalResponseNormalization& ell = layer.local_response_normalization();

      DataType lrn_alpha = ell.lrn_alpha();
      DataType lrn_beta = ell.lrn_beta();
      DataType lrn_k = ell.lrn_k();
      int window_width = ell.window_width();
      if (layout == data_layout::MODEL_PARALLEL and master) {
        err << __FILE__ << " " << __LINE__ << " :: local_response_normalization "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new local_response_normalization_layer<data_layout::DATA_PARALLEL>(
          comm,
          window_width,
          lrn_alpha,
          lrn_beta,
          lrn_k,
          cudnn);
      }
    }
#if 0

    //////////////////////////////////////////////////////////////////
    // LAYER: selu_dropout (regularizer)
    //////////////////////////////////////////////////////////////////
    else if (layer.has_selu_dropout()) {
      const lbann_data::SeluDropout& ell = layer.selu_dropout();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new selu_dropout<data_layout::MODEL_PARALLEL>(
          comm,
          ell.keep_prob(),
          ell.alpha(),
          ell.scale()
        );
      } else {
        d = new selu_dropout<data_layout::DATA_PARALLEL>(
          comm,
          ell.keep_prob(),
          ell.alpha(),
          ell.scale()
        );
      }
    }
#endif
    //////////////////////////////////////////////////////////////////
    // LAYER: batch_normalization
    //////////////////////////////////////////////////////////////////
    else if (layer.has_batch_normalization()) {
      const lbann_data::BatchNormalization& ell = layer.batch_normalization();
      if (layout == data_layout::MODEL_PARALLEL and master) {
        err << __FILE__ << " " << __LINE__ << " :: batch_normalization "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new batch_normalization<data_layout::DATA_PARALLEL>(
          comm,
          ell.decay(),
          ell.epsilon(),
          ell.global_stats(),
          cudnn);
      }
    }
#if 0
    //////////////////////////////////////////////////////////////////
    // LAYER: selu (activation)
    //////////////////////////////////////////////////////////////////
    else if (layer.has_selu()) {
      const lbann_data::Selu& ell = layer.selu();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new selu_layer<data_layout::MODEL_PARALLEL>(
          comm,
          ell.alpha(),
          ell.scale()
        );
      } else {
        d = new selu_layer<data_layout::DATA_PARALLEL>(
          comm,
          ell.alpha(),
          ell.scale()
        );
      }
    }
#endif
    //////////////////////////////////////////////////////////////////
    // LAYER: tanh
    //////////////////////////////////////////////////////////////////
    else if (layer.has_tanh()) {
      //const lbann_data::Tanh& ell = layer.tanh();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new tanh_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new tanh_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: softplus
    //////////////////////////////////////////////////////////////////
    else if (layer.has_softplus()) {
      //const lbann_data::Softplus& ell = layer.softplus();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new softplus_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new softplus_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: smooth_relu
    //////////////////////////////////////////////////////////////////
    else if (layer.has_smooth_relu()) {
      //const lbann_data::SmoothRelu& ell = layer.smooth_relu();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new smooth_relu_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new smooth_relu_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: leaky_relu
    //////////////////////////////////////////////////////////////////
    else if (layer.has_leaky_relu()) {
      const lbann_data::LeakyRelu& ell = layer.leaky_relu();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new leaky_relu_layer<data_layout::MODEL_PARALLEL>(
          comm,
          ell.leak()
        );
      } else {
        d = new leaky_relu_layer<data_layout::DATA_PARALLEL>(
          comm,
          ell.leak()
        );
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: id
    //////////////////////////////////////////////////////////////////
    else if (layer.has_id()) {
      //const lbann_data::ID& ell = layer.id();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new identity_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new identity_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: elu
    //////////////////////////////////////////////////////////////////
    else if (layer.has_elu()) {
      const lbann_data::ELU& ell = layer.elu();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new elu_layer<data_layout::MODEL_PARALLEL>(
          comm,
          ell.alpha()
        );
      } else {
        d = new elu_layer<data_layout::DATA_PARALLEL>(
          comm,
          ell.alpha()
        );
      }
    }

    //#####

    //////////////////////////////////////////////////////////////////
    // LAYER: atan
    //////////////////////////////////////////////////////////////////
    else if (layer.has_atan()) {
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new atan_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new atan_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: bent_identity
    //////////////////////////////////////////////////////////////////
    else if (layer.has_bent_identity()) {
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new bent_identity_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new bent_identity_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: exponential
    //////////////////////////////////////////////////////////////////
    else if (layer.has_exponential()) {
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new exponential_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new exponential_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: swish
    //////////////////////////////////////////////////////////////////
    else if (layer.has_swish()) {
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new swish_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new swish_layer<data_layout::DATA_PARALLEL>(comm);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: dropout
    //////////////////////////////////////////////////////////////////
    else if (layer.has_dropout()) {
      const lbann_data::Dropout& ell = layer.dropout();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new dropout<data_layout::MODEL_PARALLEL>(
          comm,
          ell.keep_prob());
      } else {
        d = new dropout<data_layout::DATA_PARALLEL>(
          comm,
          ell.keep_prob());
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: softmax
    //////////////////////////////////////////////////////////////////
    else if (layer.has_softmax()) {
      //const lbann_data::Softmax& ell = layer.softmax();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new softmax_layer<data_layout::MODEL_PARALLEL>(comm, cudnn);
      } else {
        d = new softmax_layer<data_layout::DATA_PARALLEL>(comm,cudnn);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: target_partitioned_minibatch
    //////////////////////////////////////////////////////////////////
    else if (layer.has_target_partitioned_minibatch()) {
      const lbann_data::TargetPartitionedMinibatch& ell = layer.target_partitioned_minibatch();
      if (layout == data_layout::MODEL_PARALLEL and master) {
        err << __FILE__ << " " << __LINE__ << " :: target_layer_partitioned_minibatch "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new  target_layer_partitioned_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          nullptr,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: target_distributed_minibatch
    //////////////////////////////////////////////////////////////////
    else if (layer.has_target_distributed_minibatch()) {
      const lbann_data::TargetDistributedMinibatch& ell = layer.target_distributed_minibatch();
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new  target_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(
          comm,
          nullptr,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      } else {
        d = new  target_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          nullptr,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      }
    }

    //////////////////////////////////////////////////////////////////
    // ERROR
    //////////////////////////////////////////////////////////////////
    else {
      if (master) {
        err << __FILE__ << " " << __LINE__
            << " :: unknown or unsupported layer type";
        throw lbann_exception(err.str());
      }
    }

    // Set layer name
    std::string layer_name = layer.name();
    if (!layer_name.empty()) {
      d->set_name(layer_name);
    } else {
      layer_name = d->get_name();
    }
    if (master and layer_is_in_model(layer_name)) {
      err << __FILE__ << " " << __LINE__
          << " :: layer name " << layer_name << " is not unique" ;
      throw lbann_exception(err.str());
    }

    // Add layer to model
    model->add_layer(d);
    model_layers[layer_name] = d;
    model_layer_names.push_back(layer_name);

  }

  setup_pointers(proto_layers, model, master);

}

lbann_summary * construct_summarizer(const lbann_data::Model &m, lbann_comm *comm) {
  lbann_summary *summary = nullptr;
  bool master = comm->am_world_master();
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);
    if (callback.has_summary()) {
      const lbann_data::CallbackSummary& c = callback.summary();
      if (master) {
        std::cout << "constructing summarizer with dir: " << c.dir() << std::endl;
      }

      //check to see if directory exists
      struct stat sb;
      if (! ( stat(c.dir().c_str(), &sb) == 0 && S_ISDIR(sb.st_mode) )) {
        if (master) {
          throw lbann_exception(
            std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
            "summary directory " + c.dir() + " does not exist");
        }
      }
      summary = new lbann_summary(c.dir(), comm);
    }
  }
  return summary;
}


void choose_imcomm_callback_weights(lbann_comm *comm,
                                    const lbann_data::Model& m,
                                    std::unordered_set<std::string> &include_list,
                                    std::unordered_set<std::string> &exclude_list) {
  const bool master = comm->am_world_master();
  const int num_weights = m.weights_size();
  for (int j=0; j<num_weights; j++) {
    const lbann_data::Weights& w = m.weights(j);
    switch (w.imcomm()) {
      case lbann_data::Imcomm::DEFAULT :
        break;
      case lbann_data::Imcomm::EXCLUDE :
        exclude_list.insert(w.name());
        if (master) {
          std::cout << "EXPLICITLY EXCLUDING: " << w.name() << std::endl;
        }
        break;
      case lbann_data::Imcomm::INCLUDE :
        include_list.insert(w.name());
        if (master) {
          std::cout << "EXPLICITLY INCLUDING: " << w.name() << std::endl;
        }
        break;
      //todo TODO need error checking here
      case lbann_data::Imcomm::Imcomm_INT_MIN_SENTINEL_DO_NOT_USE_ :
        break;
      case lbann_data::Imcomm::Imcomm_INT_MAX_SENTINEL_DO_NOT_USE_ :
        break;
    }
  }

}

void init_callbacks(
  lbann_comm *comm,
  lbann::model *model,
  std::map<execution_mode, lbann::generic_data_reader *>& data_readers,
  const lbann_data::LbannPB& p)
{
  std::stringstream err;
  bool master = comm->am_world_master();

  const lbann_data::Model& m = p.model();
  if (master) std::cerr << std::endl << "starting init_callbacks; size: " << m.callback_size() << std::endl;

  //the same summarizer is passed to all call backs that take a summarizer;
  //construct_summarizer returns this summarizer, which may be a nullptr
  lbann_summary *summarizer = construct_summarizer(m, comm);

  //loop over the callbacks
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);

#if 0
    //////////////////////////////////////////////////////////////////
    // CALLBACK: ltfb
    //////////////////////////////////////////////////////////////////
    if (callback.has_ltfb()) {
      const lbann_data::CallbackLTFB &c = callback.ltfb();
      auto *ltfb_cb = new lbann_callback_ltfb(c.round_size(), summarizer);
      model->add_callback(ltfb_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: save_images
    //////////////////////////////////////////////////////////////////
    if (callback.has_save_images()) {
      const lbann_data::CallbackSaveImages& c = callback.save_images();
      std::string image_dir = c.image_dir();
      std::string extension = c.extension();
      generic_data_reader *reader = data_readers[execution_mode::training];
      lbann_callback_save_images *image_cb = new lbann_callback_save_images(reader, image_dir, extension);
      model->add_callback(image_cb);
    }
#endif
    //////////////////////////////////////////////////////////////////
    // CALLBACK: print
    //////////////////////////////////////////////////////////////////
    if (callback.has_print()) {
      const lbann_data::CallbackPrint& c = callback.print();
      if (master) {
        std::cout << "adding print callback" << std::endl;
      }
      auto *print_cb = new lbann_callback_print(c.interval());
      model->add_callback(print_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: timer
    //////////////////////////////////////////////////////////////////
    if (callback.has_timer()) {
      auto *timer_cb = new lbann_callback_timer(summarizer);
      model->add_callback(timer_cb);
    }
#if 0
    //////////////////////////////////////////////////////////////////
    // CALLBACK: summary
    //////////////////////////////////////////////////////////////////
    if (callback.has_summary()) {
      const lbann_data::CallbackSummary& c = callback.summary();
      auto *summary_cb = new lbann_callback_summary(summarizer, c.batch_interval(), c.mat_interval());
      model->add_callback(summary_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_weights
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_weights()) {
      const lbann_data::CallbackDumpWeights& c = callback.dump_weights();
      if (master) {
        std::cout << "adding dump weights callback with basename: " << c.basename()
                  << " and interval: " << c.interval() << std::endl;
      }
      lbann_callback_dump_weights *weights_cb = new lbann_callback_dump_weights(c.basename(), c.interval());
      model->add_callback(weights_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_activations
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_activations()) {
      const lbann_data::CallbackDumpActivations& c = callback.dump_activations();
      if (master) {
        std::cout << "adding dump activations callback with basename: " << c.basename()
                  << " and interval: " << c.interval() << std::endl;
      }
      lbann_callback_dump_activations *activations_cb = new lbann_callback_dump_activations(c.basename(), c.interval());
      model->add_callback(activations_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_gradients
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_gradients()) {
      const lbann_data::CallbackDumpGradients& c = callback.dump_gradients();
      if (master) {
        std::cout << "adding dump gradients callback with basename: " << c.basename()
                  << " and interval: " << c.interval() << std::endl;
      }
      lbann_callback_dump_gradients *gradients_cb = new lbann_callback_dump_gradients(c.basename(), c.interval());
      model->add_callback(gradients_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_mb_indices
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_mb_indices()) {
      const lbann_data::CallbackDumpMBIndices& c = callback.dump_mb_indices();
      if (master) {
        std::cout << "adding dump I/O callback with basename: " << c.basename()
                  << " and interval: " << c.interval() << std::endl;
      }
      lbann_callback_dump_minibatch_sample_indices *mb_indices_cb = new lbann_callback_dump_minibatch_sample_indices(c.basename(), c.interval());
      model->add_callback(mb_indices_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: check_dataset
    //////////////////////////////////////////////////////////////////
    if (callback.has_check_dataset()) {
      //const lbann_data::CallbackCheckDataset& c = callback.check_dataset();
      if (master) {
        std::cout << "adding callback to check the dataset" << std::endl;
      }
      auto *check_dataset_cb = new lbann_callback_check_dataset();
      model->add_callback(check_dataset_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: disp_io_stats
    //////////////////////////////////////////////////////////////////
    if (callback.has_disp_io_stats()) {
      const lbann_data::CallbackDispIOStats& c = callback.disp_io_stats();
      std::stringstream s(c.layers());
      std::unordered_set<Layer*> which;
      std::string a;
      while (s >> a) {
        if (master and not layer_is_in_model(a)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback disp_io_stats: could not find layer " << a;
          throw lbann_exception(err.str());
        }
        which.insert(model_layers[a]);
        if (master) {
          std::cout << "adding display I/O stats callback for layer " << a;
        }
      }
      lbann_callback_io *io_cb = new lbann_callback_io(which);
      model->add_callback(io_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: imcomm
    //////////////////////////////////////////////////////////////////
    if (callback.has_imcomm()) {
      const lbann_data::CallbackImComm& c = callback.imcomm();
      if (master) {
        std::cout << "adding imcomm callback\n";
      }
      std::unordered_set<std::string> include_list, exclude_list;
      choose_imcomm_callback_weights(comm, m, include_list, exclude_list);
      if (c.all_optimizers()) {
        for (auto it : model_weights) {
          std::string name = it.second->get_name();
          if (exclude_list.find(name) == exclude_list.end()) {
            if (master) {
              std::cout << "ADDING to IMCOMM: " << name << std::endl;
            }
            include_list.insert(name);
          } else {
            if (master) {
              std::cout << "WOULD ADD TO IMCOMM, but was explicitly excluded: "
                        << name << std::endl;
            }
          }
        }
      }
      std::unordered_set<weights*> weights_list;
      for (std::string name : include_list) {
        if (master && !weights_are_in_model(name)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback imcomm: could not find " << name;
          throw lbann_exception(err.str());
        }
        weights_list.insert(model_weights[name]);
      }
      lbann_callback_imcomm::comm_type c_type = get_comm_type(c.intermodel_comm_method(), master);
      lbann_callback_imcomm *im = new lbann_callback_imcomm(c_type, weights_list, summarizer);
      model->add_callback(im);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: step_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_step_learning_rate()) {
      const lbann_data::CallbackStepLearningRate &c = callback.step_learning_rate();
      std::stringstream s(c.weights());
      std::unordered_set<weights*> weights_list;
      std::string name;
      while (s >> name) {
        if (master && !weights_are_in_model(name)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback step_learning_rate: could not find " << name;
          throw lbann_exception(err.str());
        }
        weights_list.insert(model_weights[name]);
      }
      lbann_callback_step_learning_rate *learn
        = new lbann_callback_step_learning_rate(c.step(), c.amt(), weights_list);
      if (master) {
        std::cout << "adding step learning rate callback\n";
      }
      model->add_callback(learn);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: adaptive_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_adaptive_learning_rate()) {
      const lbann_data::CallbackAdaptiveLearningRate &c = callback.adaptive_learning_rate();
      std::stringstream s(c.weights());
      std::unordered_set<weights*> weights_list;
      std::string name;
      while (s >> name) {
        if (master && !weights_are_in_model(name)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback adaptive_learning_rate: could not find " << name;
          throw lbann_exception(err.str());
        }
        weights_list.insert(model_weights[name]);
      }
      lbann_callback_adaptive_learning_rate *learn
        = new lbann_callback_adaptive_learning_rate(c.patience(), c.amt(), weights_list);
      model->add_callback(learn);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: debug
    //////////////////////////////////////////////////////////////////
    if (callback.has_debug()) {
      const lbann_data::CallbackDebug& c = callback.debug();
      if (master) {
        std::cout << "adding debugging callback for phase: " << c.phase() << std::endl;
      }
      lbann_callback_debug *debug_cb = nullptr;
      if(c.phase() == "train" || c.phase() == "training") {
        debug_cb = new lbann_callback_debug(execution_mode::training, summarizer);
      } else if (c.phase() == "validation") {
        debug_cb = new lbann_callback_debug(execution_mode::validation, summarizer);
      } else if (c.phase() == "test" || c.phase() == "testing") {
        debug_cb = new lbann_callback_debug(execution_mode::testing, summarizer);
      } else {
        debug_cb = new lbann_callback_debug();
      }
      model->add_callback(debug_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: debug_io
    //////////////////////////////////////////////////////////////////
    if (callback.has_debug_io()) {
      const lbann_data::CallbackDebugIO& c = callback.debug_io();
      if (master) {
        std::cout << "adding debugging I/O callback for phase: " << c.phase() << std::endl;
      }
      lbann_callback_debug_io *debug_cb = nullptr;
      if(c.phase() == "train" || c.phase() == "training") {
        debug_cb = new lbann_callback_debug_io(execution_mode::training, c.lvl());
      } else if (c.phase() == "validate" || c.phase() == "validation") {
        debug_cb = new lbann_callback_debug_io(execution_mode::validation, c.lvl());
      } else if (c.phase() == "test" || c.phase() == "testing") {
        debug_cb = new lbann_callback_debug_io(execution_mode::testing, c.lvl());
      } else {
        debug_cb = new lbann_callback_debug_io();
      }
      model->add_callback(debug_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: check_small
    //////////////////////////////////////////////////////////////////
    if (callback.has_check_small()) {
      if (master) {
        std::cout << "adding check_small callback" << std::endl;
      }
      auto *checksmall_cb = new lbann_callback_checksmall();
      model->add_callback(checksmall_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: check_nan
    //////////////////////////////////////////////////////////////////
    if (callback.has_check_nan()) {
      if (master) {
        std::cout << "adding check_nan callback" << std::endl;
      }
      auto *checknan_cb = new lbann_callback_checknan();
      model->add_callback(checknan_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: hang
    //////////////////////////////////////////////////////////////////
    if (callback.has_hang()) {
      const lbann_data::CallbackHang& c = callback.hang();
      int rank_to_hang = c.rank();
      if (master) {
        if (rank_to_hang == -1) {
          std::cout << "*** HANGING EVERY RANK IN HANG CALLBACK ***" <<
                    std::endl;
        } else {
          std::cout << "*** HANGING RANK " << rank_to_hang <<
                    " IN HANG CALLBACK ***" << std::endl;
        }
      }
      auto *hang_cb = new lbann_callback_hang(rank_to_hang);
      model->add_callback(hang_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: drop_fixed_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_drop_fixed_learning_rate()) {
      const lbann_data::CallbackDropFixedLearningRate& c =
        callback.drop_fixed_learning_rate();
      if (master) {
        std::cout << "adding drop_fixed_learning_rate callback" << std::endl;
      }
      std::stringstream s(c.weights());
      std::unordered_set<weights*> weights_list;
      std::string name;
      while (s >> name) {
        if (master && !weights_are_in_model(name)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback drop_learning_rate: could not find " << name;
          throw lbann_exception(err.str());
        }
        weights_list.insert(model_weights[name]);
      }
      std::vector<int64_t> drop_epochs;
      for (int i = 0; i < c.drop_epoch_size(); ++i) {
        drop_epochs.push_back(c.drop_epoch(i));
      }
      lbann_callback_drop_fixed_learning_rate *dflr = new
      lbann_callback_drop_fixed_learning_rate(
        drop_epochs, c.amt(), weights_list);
      model->add_callback(dflr);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: linear_growth_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_linear_growth_learning_rate()) {
      const lbann_data::CallbackLinearGrowthLearningRate& c =
        callback.linear_growth_learning_rate();
      if (master) {
        std::cout << "adding linear_growth_learning_rate callback" << std::endl;
      }
      std::stringstream s(c.weights());
      std::unordered_set<weights*> weights_list;
      std::string name;
      while (s >> name) {
        if (master && !weights_are_in_model(name)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback linear_growth_learning_rate: could not find " << name;
          throw lbann_exception(err.str());
        }
        weights_list.insert(model_weights[name]);
      }
      lbann_callback_linear_growth_learning_rate *lglr = new
      lbann_callback_linear_growth_learning_rate(
        c.target(), c.num_epochs(), c.delay(), weights_list);
      model->add_callback(lglr);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: profiler
    //////////////////////////////////////////////////////////////////
    if (callback.has_profiler()) {
      //const lbann_data::CallbackProfiler& c = callback.profiler();
      if (master) {
        std::cout << "adding profiler callback" << std::endl;
      }
      auto *profiler_cb = new lbann_callback_profiler();
      model->add_callback(profiler_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: step_minibatch
    //////////////////////////////////////////////////////////////////
    if (callback.has_step_minibatch()) {
      const lbann_data::CallbackStepMinibatch& c = callback.step_minibatch();
      if (master) {
        std::cout << "adding step_minibatch callback, start=" <<
          c.starting_mbsize() << ", step=" << c.step() << " ramp=" <<
          c.ramp_time() << std::endl;
      }
      auto *step_mb_cb = new
        lbann_callback_step_minibatch(c.starting_mbsize(), c.step(),
                                      c.ramp_time());
      model->add_callback(step_mb_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: minibatch_schedule
    //////////////////////////////////////////////////////////////////
    if (callback.has_minibatch_schedule()) {
      const lbann_data::CallbackMinibatchSchedule& c =
        callback.minibatch_schedule();
      if (master) {
        std::cout << "adding minibatch_schedule callback" << std::endl;
      }
      std::vector<lbann_callback_minibatch_schedule::minibatch_step> steps;
      for (int i = 0; i < c.step_size(); ++i) {
        const lbann_data::MinibatchScheduleStep& step = c.step(i);
        steps.emplace_back(step.epoch(), step.mbsize(), step.lr(),
                           step.ramp_time());
      }
      lbann_callback_minibatch_schedule *mb_sched = new
        lbann_callback_minibatch_schedule(c.starting_mbsize(), steps);
      model->add_callback(mb_sched);
    }
#endif
    //////////////////////////////////////////////////////////////////
    // CALLBACK: gradient_check
    //////////////////////////////////////////////////////////////////
    if (callback.has_gradient_check()) {
      const lbann_data::CallbackGradientCheck& c = callback.gradient_check();
      if (master) {
        std::cout << "adding gradient_check callback" << std::endl;
      }
      auto *gradient_check_cb = new
      lbann_callback_gradient_check(c.step_size(), c.verbose(), c.fail_on_error());
      model->add_callback(gradient_check_cb);
    }
#if 0
    //////////////////////////////////////////////////////////////////
    // CALLBACK: optimizerwise_adaptive_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_optimizerwise_adaptive_learning_rate()) {
      const lbann_data::CallbackOptimizerwiseAdaptiveLearningRate& c =
        callback.optimizerwise_adaptive_learning_rate();
      if (master) {
        std::cout << "adding optimizerwise_adaptive_learning_rate callback" <<
          " with scale=" << c.scale() << std::endl;
      }
      std::stringstream s(c.weights());
      std::unordered_set<weights*> weights_list;
      std::string name;
      while (s >> name) {
        if (master && !weights_are_in_model(name)) {
          err << __FILE__ << " " << __LINE__
              << " :: callback optimizerwise_adaptive_learning_rate: could not find " << name;
          throw lbann_exception(err.str());
        }
        weights_list.insert(model_weights[name]);
      }
      lbann_callback_optimizerwise_adaptive_learning_rate *owalr_cb = new
        lbann_callback_optimizerwise_adaptive_learning_rate(c.scale(), weights_list);
      model->add_callback(owalr_cb);
    }


    //////////////////////////////////////////////////////////////////
    // CALLBACK: checkpoint
    //////////////////////////////////////////////////////////////////
    if (callback.has_checkpoint()) {
      const lbann_data::CallbackCheckpoint& c = callback.checkpoint();
      if (master) {
        std::cout << "checkpoint saving on interval <epoch:" << c.checkpoint_epochs() << " steps:" << c.checkpoint_steps() << " secs:" << c.checkpoint_secs()  
	          << "> to dir: " << c.checkpoint_dir() << std::endl;
      }
      lbann_callback_checkpoint *checkpoint_cb = new
        lbann_callback_checkpoint(c.checkpoint_dir(), c.checkpoint_epochs(), c.checkpoint_steps(), c.checkpoint_secs(), c.checkpoint_per_rank());
      model->add_callback(checkpoint_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: save_model
    //////////////////////////////////////////////////////////////////
    if (callback.has_save_model()) {
      const lbann_data::CallbackSaveModel& c = callback.save_model();
      std::string dir = c.dir();
      std::string extension = c.extension();
      lbann_callback_save_model *model_cb = new lbann_callback_save_model(dir, extension);
      model->add_callback(model_cb);
    }

#endif
  }

}

objective_function *init_objective_function(lbann_data::ObjectiveFunction obj_fn_params) {
  objective_function *obj_fn = new objective_function();
  for (int j=0; j<obj_fn_params.mean_squared_error_size(); j++) {
    const lbann_data::MeanSquaredError &params = obj_fn_params.mean_squared_error(j);
    obj_fn->add_term(new mean_squared_error_loss(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.mean_absolute_deviation_size(); j++) {
    const lbann_data::MeanAbsoluteDeviation &params = obj_fn_params.mean_absolute_deviation(j);
    obj_fn->add_term(new mean_absolute_deviation_loss(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.cross_entropy_size(); j++) {
    const lbann_data::CrossEntropy &params = obj_fn_params.cross_entropy(j);
    obj_fn->add_term(new cross_entropy(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.binary_cross_entropy_size(); j++) {
    const lbann_data::BinaryCrossEntropy &params = obj_fn_params.binary_cross_entropy(j);
    obj_fn->add_term(new binary_cross_entropy(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.cross_entropy_with_uncertainty_size(); j++) {
    const lbann_data::CrossEntropyWithUncertainty &params = obj_fn_params.cross_entropy_with_uncertainty(j);
    obj_fn->add_term(new cross_entropy_with_uncertainty(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.geom_negloglike_size(); j++) {
    const lbann_data::GeomNegLogLike &params = obj_fn_params.geom_negloglike(j);
    obj_fn->add_term(new geom_negloglike(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.poisson_negloglike_size(); j++) {
    const lbann_data::PoissonNegLogLike &params = obj_fn_params.poisson_negloglike(j);
    obj_fn->add_term(new poisson_negloglike(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.polya_negloglike_size(); j++) {
    const lbann_data::PolyaNegLogLike &params = obj_fn_params.polya_negloglike(j);
    obj_fn->add_term(new polya_negloglike(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.l1_weight_regularization_size(); j++) {
    const lbann_data::L1WeightRegularization &params = obj_fn_params.l1_weight_regularization(j);
    obj_fn->add_term(new l1_weight_regularization(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.l2_weight_regularization_size(); j++) {
    const lbann_data::L2WeightRegularization &params = obj_fn_params.l2_weight_regularization(j);
    obj_fn->add_term(new l2_weight_regularization(params.scale_factor()));
  }
  for (int j=0; j<obj_fn_params.group_lasso_weight_regularization_size(); j++) {
    const lbann_data::GroupLassoWeightRegularization &params = obj_fn_params.group_lasso_weight_regularization(j);
    obj_fn->add_term(new group_lasso_weight_regularization(params.scale_factor()));
  }
  return obj_fn;
}

model *init_model(lbann_comm *comm, optimizer *default_optimizer, const lbann_data::LbannPB& p)
{
  std::stringstream err;
  bool master = comm->am_world_master();

  //sequential_model *model = 0;
  model *model = nullptr;

  const lbann_data::Model& m = p.model();
  const std::string name = m.name();
  uint mini_batch_size = m.mini_batch_size();

  //instantiate the objective function
  objective_function *obj_fn = init_objective_function(m.objective_function());

  //instantiate the network; layers will be added in a separate function call
  if (name == "sequential_model") {
    model = new sequential_model(comm, mini_batch_size, obj_fn, default_optimizer);
    if (master) std::cout << "instantiating sequential_model\n";
  } 
#if 0
else if (name == "directed_acyclic_graph_model") {
    model = new directed_acyclic_graph_model(comm, mini_batch_size, obj_fn, default_optimizer);
    if (master) std::cout << "instantiating directed_acyclic_graph_model\n";
  } else if (name == "recurrent_model") {
    const lbann_data::Model::Recurrent& recurrent = m.recurrent();
    model = new recurrent_model(comm, mini_batch_size, obj_fn, default_optimizer, recurrent.unroll_depth());
    if (master) std::cout << "instantiating recurrent_model\n";
  } else if(name == "siamese_model") {
    if (m.has_siamese()) {
      const lbann_data::Model::Siamese& siamese = m.siamese();
      model = new siamese_model(comm, mini_batch_size, obj_fn, default_optimizer, siamese.num_heads());
    } else {
      err << __FILE__ << " " << __LINE__
          << " :: init_model() - " << name << " needs definition" << std::endl;
      throw lbann_exception(err.str());
    }
    if (master) std::cout << "instantiating siamese_model\n";
  } else if (name == "greedy_layerwise_autoencoder") {
    model = new greedy_layerwise_autoencoder(comm, mini_batch_size, obj_fn, default_optimizer);
    if (master) std::cout << "instantiating greedy_layerwise_autoencoder\n";
  }
#endif
  else {
    if (master) {
      err << __FILE__ << " " << __LINE__
          << " :: init_model() - unknown model name: " << name << std::endl
          << "; should be one of: sequential_model, dag_model, greedy_layerwise_autoencoder";
      throw lbann_exception(err.str());
    }
  }

  //get data layout
  const data_layout layout = get_data_layout(m.data_layout());
  if (master and layout == data_layout::invalid) {
    err << __FILE__ << " " << __LINE__ << " :: "
        << "model has invalid data layout " << m.data_layout();
    throw lbann_exception(err.str());
  }

  //add the metrics
  for (int j=0; j<m.metric_size(); j++) {
    const lbann_data::Metric &metric = m.metric(j);
    if (metric.has_categorical_accuracy()) {
      model->add_metric(new categorical_accuracy_metric(comm));
    }
    if (metric.has_top_k_categorical_accuracy()) {
      const lbann_data::TopKCategoricalAccuracy &a = metric.top_k_categorical_accuracy();
      model->add_metric(new top_k_categorical_accuracy_metric(a.top_k(), comm));
    }
    if (metric.has_mean_squared_error()) {
      model->add_metric(new mean_squared_error_metric(comm));
    }
    if (metric.has_mean_absolute_deviation()) {
      model->add_metric(new mean_absolute_deviation_metric(comm));
    }
    if (metric.has_pearson_correlation()) {
      model->add_metric(new pearson_correlation_metric(comm));
    }
  }

  //set checkpoint values
  //model->set_checkpoint_dir(m.checkpoint_dir());
  //model->set_checkpoint_epochs(m.checkpoint_epochs());
  //model->set_checkpoint_steps(m.checkpoint_steps());
  //model->set_checkpoint_secs(m.checkpoint_secs());

  return model;
}

optimizer *init_default_optimizer(lbann_comm *comm,
                                  cudnn::cudnn_manager *cudnn,
                                  const lbann_data::LbannPB& params)
{
  optimizer *opt = nullptr;
  const lbann_data::Optimizer &opt_params = params.optimizer();
  if (opt_params.has_sgd()) {
    const lbann_data::Sgd &sgd_params = opt_params.sgd();
    opt = new sgd(comm,
                  sgd_params.learn_rate(),
                  sgd_params.momentum(),
                  sgd_params.nesterov());
  }
  if (opt_params.has_adagrad()) {
    const lbann_data::Adagrad &adagrad_params = opt_params.adagrad();
    opt = new adagrad(comm,
                      adagrad_params.learn_rate(),
                      adagrad_params.eps());
  }
  if (opt_params.has_rmsprop()) {
    const lbann_data::Rmsprop &rmsprop_params = opt_params.rmsprop();
    opt = new rmsprop(comm,
                      rmsprop_params.learn_rate(),
                      rmsprop_params.decay_rate(),
                      rmsprop_params.eps());
  }
  if (opt_params.has_adam()) {
    const lbann_data::Adam &adam_params = opt_params.adam();
    opt = new adam(comm,
                   adam_params.learn_rate(),
                   adam_params.beta1(),
                   adam_params.beta2(),
                   adam_params.eps());
  }
  if (opt_params.has_hypergradient_adam()) {
    const lbann_data::HypergradientAdam &hypergradient_adam_params = opt_params.hypergradient_adam();
    opt = new hypergradient_adam(comm,
                                 hypergradient_adam_params.init_learning_rate(),
                                 hypergradient_adam_params.hyper_learning_rate(),
                                 hypergradient_adam_params.beta1(),
                                 hypergradient_adam_params.beta2(),
                                 hypergradient_adam_params.eps());
  }

  return opt;
}


void init_data_readers(bool master, const lbann_data::LbannPB& p, std::map<execution_mode, generic_data_reader *>& data_readers)
{
  std::stringstream err;

  const lbann_data::DataReader & d_reader = p.data_reader();
  int size = d_reader.reader_size();

  for (int j=0; j<size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);
    // This is a temporary measure until we individually setup data reader specific preprocessors
    bool set_up_generic_preprocessor = true;

    const std::string& name = readme.name();

    const bool shuffle = readme.shuffle();

    generic_data_reader *reader = nullptr;
    generic_data_reader *reader_validation = nullptr;

    if ((name == "imagenet_org") || (name == "mnist") || (name == "cifar10")) {
      init_org_image_data_reader(readme, master, reader);
      set_up_generic_preprocessor = false;
    } else if ((name == "imagenet") || (name == "imagenet_single") || (name == "imagenet_patches")) {
      init_image_data_reader(readme, master, reader);
      set_up_generic_preprocessor = false;
    } else if (name == "nci") {
      reader = new data_reader_nci(shuffle);
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
    } else if (name == "numpy") {
      auto* reader_numpy = new numpy_reader(shuffle);
      reader_numpy->set_has_labels(!readme.disable_labels());
      reader_numpy->set_has_responses(!readme.disable_responses());
      reader = reader_numpy;
    } else if (name == "pilot2_molecular_reader") {
      pilot2_molecular_reader* reader_pilot2_molecular = new pilot2_molecular_reader(readme.num_neighbors(), readme.max_neighborhood(), shuffle);
      reader = reader_pilot2_molecular;
    } else if (name == "merge_samples" || name == "merge_features") {
      //TODO: verify how much of wildcard conflict with label file, label file should be loaded separately
      auto paths = glob(readme.data_file_pattern());
      std::vector<generic_data_reader*> npy_readers;
      for (const auto path : paths) {
        if(master) { std::cout << "Loading file: " << path << std::endl; }
        if (readme.format() == "numpy") {
          auto *reader_numpy = new numpy_reader(false);
          reader_numpy->set_data_filename(path);
          reader_numpy->set_has_labels(!readme.disable_labels());
          reader_numpy->set_has_responses(!readme.disable_responses());
          npy_readers.push_back(reader_numpy);
        } else if (readme.format() == "pilot2_molecular_reader") {
          pilot2_molecular_reader* reader_pilot2_molecular = new pilot2_molecular_reader(readme.num_neighbors(), readme.max_neighborhood(), shuffle);
          reader_pilot2_molecular->set_data_filename(path);
          npy_readers.push_back(reader_pilot2_molecular);
        } else if (readme.format() == "csv") {
          auto* reader_csv = new csv_reader(shuffle);
          reader_csv->set_data_filename(path);
          reader_csv->set_label_col(readme.label_col());
          reader_csv->set_response_col(readme.response_col());
          reader_csv->disable_labels(readme.disable_labels());
          reader_csv->enable_responses(readme.disable_responses());
          reader_csv->set_separator(readme.separator()[0]);
          reader_csv->set_skip_cols(readme.skip_cols());
          reader_csv->set_skip_rows(readme.skip_rows());
          reader_csv->set_has_header(readme.has_header());
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
      }else {
        //create label file
        auto* label_csv = new csv_reader(shuffle); 
        label_csv->set_data_filename(readme.label_filename());
        label_csv->disable_labels(false); 
        label_csv->set_has_header(readme.has_header()); //use same as parent file
        label_csv->set_label_col(0); //assume there is only one label file and the column and is label column
        data_reader_merge_features* merged_features = new data_reader_merge_features(npy_readers,label_csv, shuffle);
        reader = merged_features;
      }
  
    } else if (name == "synthetic") {
      reader = new data_reader_synthetic(readme.num_samples(), readme.num_features(), shuffle);
    } else if (name == "ascii") {
      reader = new ascii_reader(p.model().recurrent().unroll_depth(), shuffle);
    } else {
      if (master) {
        err << __FILE__ << " " << __LINE__ << " :: unknown name for data reader: "
            << name;
        throw lbann_exception(err.str());
      }
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

    reader->set_absolute_sample_count( readme.absolute_sample_count() );
    reader->set_use_percent( readme.percent_of_data_to_use() );
    reader->set_first_n( readme.first_n() );

    if (set_up_generic_preprocessor) {
      init_generic_preprocessor(readme, master, reader);
    }

    if (readme.role() == "train") {
      reader->set_role("train");
    } else if (readme.role() == "test") {
      reader->set_role("test");
    } else {
      reader->set_role("error");
    }
    if (readme.role() == "train") {
      reader->set_validation_percent( readme.validation_percent() );
    }

    reader->set_master(master);

    reader->load();

    if (readme.role() == "train") {
      data_readers[execution_mode::training] = reader;
    } else if (readme.role() == "test") {
      data_readers[execution_mode::testing] = reader;
    }

    if (readme.role() == "train") {
      if (name == "mnist") {
        reader_validation = new mnist_reader(shuffle);
        (*(mnist_reader *)reader_validation) = (*(mnist_reader *)reader);
      } else if (name == "imagenet_org") {
        reader_validation = new imagenet_reader_org(shuffle);
        (*(imagenet_reader_org *)reader_validation) = (*(imagenet_reader_org *)reader);
      } else if (name == "imagenet") {
        reader_validation = new imagenet_reader(*dynamic_cast<const imagenet_reader*>(reader));
      } else if (name == "imagenet_single") {
        reader_validation = new imagenet_reader_single(*dynamic_cast<const imagenet_reader_single*>(reader));
      } else if (name == "imagenet_patches") {
        reader_validation = new imagenet_reader_patches(*dynamic_cast<const imagenet_reader_patches*>(reader));
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
        /*
        } else if (name == "synthetic") {
        reader_validation = new data_reader_synthetic(shuffle);
        */
      } else if (name == "ascii") {
        reader_validation = new ascii_reader(p.model().recurrent().unroll_depth(), shuffle);
        (*(ascii_reader *)reader_validation) = (*(ascii_reader *)reader);
      }

      reader_validation->swap_role("validate");
      reader_validation->use_unused_index_set();

      if (master) {
        size_t num_train = reader->get_num_data();
        size_t num_validate = reader_validation->get_num_data();
        double validate_percent = ((double) num_validate / (double) (num_train+num_validate))*100.0;
        double train_percent = ((double) num_train / (double) (num_train+num_validate))*100.0;
        std::cout << "Training using " << train_percent << "% of the training data set, which is " << reader->get_num_data() << " samples." << std::endl
                  << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->get_num_data() << " samples." << std::endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }
}

void read_prototext_file(std::string fn, lbann_data::LbannPB& pb, bool master)
{
  std::stringstream err;
  int fd = open(fn.c_str(), O_RDONLY);
  if (fd == -1) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
      throw lbann_exception(err.str());
    }
  }
  auto *input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, &pb);
  if (!success) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << std::endl;
      throw lbann_exception(err.str());
    }
  }
  input->Close();
  delete input;
}

bool write_prototext_file(const char *fn, lbann_data::LbannPB& pb)
{
  int fd = open(fn, O_APPEND | O_CREAT | O_TRUNC, 0644);
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

void set_num_parallel_readers(lbann::lbann_comm *comm, lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();

  lbann_data::Model *model = p.mutable_model();

  int parallel_io = model->num_parallel_readers();
  if (parallel_io == 0) {
    if (master) {
      std::cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
        " (Limited to # Processes)" << std::endl;
    }
    parallel_io = comm->get_procs_per_model();
    model->set_num_parallel_readers(parallel_io); //adjust the prototext
  } else {
    if (master) {
      std::cout << "\tMax Parallel I/O Fetch: " << parallel_io << std::endl;
    }
  }
}

void set_data_readers_filenames(std::string which, lbann_data::LbannPB& p)
{
  options *opts = options::get();
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    if (r->role() == which) {
      std::stringstream s;
      s << "data_filedir_" << which;
      if (opts->has_string(s.str().c_str())) {
        r->set_data_filedir(opts->get_string(s.str().c_str()));
      }
      s.clear();
      s.str("");
      s << "data_filename_" << which;
      if (opts->has_string(s.str().c_str())) {
        r->set_data_filename(opts->get_string(s.str().c_str()));
      }
      s.clear();
      s.str("");
      s << "label_filename_" << which;
      if (opts->has_string(s.str().c_str())) {
        r->set_label_filename(opts->get_string(s.str().c_str()));
      }
    }
  }
}

void set_data_readers_percent(lbann_data::LbannPB& p)
{
  options *opts = options::get();
  double percent = opts->get_float("data_reader_percent");
  if (percent <= 0 || percent > 1.0) {
      std::stringstream err;
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

void get_cmdline_overrides(lbann::lbann_comm *comm, lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();
  std::stringstream err;

  options *opts = options::get();
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

  if (opts->has_string("dag_model")) {
    std::string sanity = model->name();
    if (sanity != "dnn") {
      err << __FILE__ << " " << __LINE__ << " :: "
          << " the current network model is: " << model->name()
          << "; you can only change the model to 'dag_model' if the current model is 'dnn'";
      throw lbann_exception(err.str());
    }
    if (master) {
      std::cout << "\nchanging model from " << model->name() << " to: dag\n\n";
    }
    model->set_name("dag_model");
  }

  if (opts->has_string("data_filedir_train") or opts->has_string("data_filename_train")
      or opts->has_string("label_filename_train")) {
    set_data_readers_filenames("train", p);
  }
  if (opts->has_string("data_filedir_test") or opts->has_string("data_filename_test")
      or opts->has_string("label_filename_test")) {
    set_data_readers_filenames("test", p);
  }
  if (opts->has_string("data_reader_percent")) {
    set_data_readers_percent(p);
  }

  if (opts->has_string("image_dir")) {
    int sz = model->callback_size();
    for (int j=0; j<sz; j++) {
      lbann_data::Callback *c = model->mutable_callback(j);
      if (c->has_save_images()) {
        lbann_data::CallbackSaveImages *i = c->mutable_save_images();
        i->set_image_dir(opts->get_string("image_dir"));
      }
    }
  }
  if (opts->has_bool("no_im_comm") and opts->get_bool("no_im_comm")) {
    int sz = model->callback_size();
    for (int j=0; j<sz; j++) {
      lbann_data::Callback *c = model->mutable_callback(j);
      if (c->has_imcomm()) {
        c->clear_imcomm();
      }
    }
  }

  if (opts->has_int("mini_batch_size")) {
    model->set_mini_batch_size(opts->get_int("mini_batch_size"));
  }
  if (opts->has_int("num_epochs")) {
    model->set_num_epochs(opts->get_int("num_epochs"));
  }
  if (opts->has_int("block_size")) {
    model->set_block_size(opts->get_int("block_size"));
  }
  if (opts->has_int("procs_per_model")) {
    model->set_procs_per_model(opts->get_int("procs_per_model"));
  }
  if (opts->has_int("num_gpus")) {
    model->set_num_gpus(opts->get_int("num_gpus"));
  }
  if (opts->has_int("num_parallel_readers")) {
    model->set_num_parallel_readers(opts->get_int("num_parallel_readers"));
  }
  if (opts->has_bool("use_cudnn")) {
    model->set_use_cudnn(opts->get_int("use_cudnn"));
  }
  if (opts->has_int("random_seed")) {
    model->set_random_seed(opts->get_int("random_seed"));
  }


  if (opts->has_string("opt")) {
    //defaults
    double learn_rate = opts->has_float("learn_rate") ? opts->get_float("learn_rate") : 0.01;
    double eps = opts->has_float("eps") ? opts->get_float("eps") : 1e-8;
    double beta1 = opts->has_float("beta1") ? opts->get_float("beta1") : 0.9;
    double beta2 = opts->has_float("beta2") ? opts->get_float("beta2") : 0.99;
    double init_learning_rate = opts->has_float("init_learning_rate") ? opts->get_float("init_learning_rate") : 0.01;
    double hyper_learning_rate = opts->has_float("hyper_learning_rate") ? opts->get_float("hyper_learning_rate") : 1e-7;
    double momentum = opts->has_float("momentum") ? opts->get_float("momentum") : 0.9;
    double decay_rate = opts->has_float("decay_rate") ? opts->get_float("decay_rate") : 0.5;
    bool nesterov = opts->has_bool("nesterov") ? opts->get_float("nesterov") : false;

    auto *opt = new lbann_data::Optimizer;

    //construct the new optimizer
    std::string opt_string = opts->get_string("opt");
    if (opt_string == "adagrad") {
      auto *a = new lbann_data::Adagrad;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      opt->set_allocated_adagrad(a);
    } else if (opt_string == "adam") {
      auto *a = new lbann_data::Adam;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      opt->set_allocated_adam(a);
    } else if (opt_string == "hypergradient_adam") {
      auto *a = new lbann_data::HypergradientAdam;
      a->set_init_learning_rate(init_learning_rate);
      a->set_hyper_learning_rate(hyper_learning_rate);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      a->set_eps(eps);
      opt->set_allocated_hypergradient_adam(a);
    } else if (opt_string == "rmsprop") {
      auto *a = new lbann_data::Rmsprop;
      a->set_learn_rate(learn_rate);
      a->set_decay_rate(decay_rate);
      a->set_eps(eps);
      opt->set_allocated_rmsprop(a);
    } else if (opt_string == "sgd") {
      if (master) std::cerr << "\n\nsetting: sgd\n\n";
      auto *a = new lbann_data::Sgd;
      a->set_learn_rate(learn_rate);
      a->set_momentum(momentum);
      a->set_decay_rate(decay_rate);
      a->set_nesterov(nesterov);
      opt->set_allocated_sgd(a);
    } else {
      err << __FILE__ << " " << __LINE__
          << " :: unknown string for --optimizer: " << opt_string
          << " should be on of: adagrad, adam, hypergradient_adam, rmsprop, sgd";
      throw lbann_exception(err.str());
    }
    p.set_allocated_optimizer(opt);
  }
}

void print_parameters(lbann::lbann_comm *comm, lbann_data::LbannPB& p)
{
  if (!comm->am_world_master()) {
    return;
  }

  const lbann_data::Model &m = p.model();

  std::cout << std::endl
            << "Running with these parameters:\n"
            << " General:\n"
            << "  datatype size:        " << sizeof(DataType) << std::endl
            << "  mini_batch_size:      " << m.mini_batch_size() << std::endl
            << "  num_epochs:           " << m.num_epochs()  << std::endl
            << "  block_size:           " << m.block_size()  << std::endl
            << "  procs_per_model:      " << m.procs_per_model()  << std::endl
            << "  num_gpus:             " << m.num_gpus()  << std::endl
            << "  num_parallel_readers: " << m.num_parallel_readers()  << std::endl
            << "  use_cudnn:            " << m.use_cudnn()  << std::endl
            << "  random_seed:          " << m.random_seed() << std::endl
            << "  data_layout:          " << m.data_layout()  << std::endl
            << "     (only used for metrics)\n"
            << "\n"
            << " Optimizer:  ";

  const lbann_data::Optimizer &o = p.optimizer();
  if (o.has_adagrad()) {
    const lbann_data::Adagrad &a = o.adagrad();
    std::cout << "  Adagrad\n"
              << "  learn_rate: " << a.learn_rate()  << std::endl
              << "  eps:        " << a.eps()  << std::endl;
  } else if (o.has_rmsprop()) {
    const lbann_data::Rmsprop &a = o.rmsprop();
    std::cout <<  "  Rmsprop\n"
              << "  learn_rate: " << a.learn_rate()  << std::endl
              << "  decay_rate: " << a.decay_rate()  << std::endl
              << "  eps:        " << a.eps()  << std::endl;
  } else if (o.has_adam()) {
    const lbann_data::Adam &a = o.adam();
    std::cout << "  Adam\n"
              << "  learn_rate: " << a.learn_rate()  << std::endl
              << "  beta1:      " << a.beta1()  << std::endl
              << "  beta2:      " << a.beta2()  << std::endl
              << "  eps:        " << a.eps()  << std::endl;
  } else if (o.has_hypergradient_adam()) {
    const lbann_data::HypergradientAdam &a = o.hypergradient_adam();
    std::cout << "  HypergradientAdam\n"
              << "  init_learning_rate:  " << a.init_learning_rate()  << std::endl
              << "  hyper_learning_rate: " << a.hyper_learning_rate()  << std::endl
              << "  beta1:               " << a.beta1()  << std::endl
              << "  beta2:               " << a.beta2()  << std::endl
              << "  eps:                 " << a.eps()  << std::endl;
  } else if (o.has_sgd()) {
    const lbann_data::Sgd &a = o.sgd();
    std::cout << "  Sgd\n"
              << "  learn_rate: " << a.learn_rate()  << std::endl
              << "  momentum:   " << a.momentum()  << std::endl
              << "  decay_rate: " << a.decay_rate()  << std::endl
              << "  nesterov:   " << a.nesterov()  << std::endl;
  }
}

void print_help(lbann::lbann_comm *comm)
{
  if (!comm->am_world_master()) {
    return;
  }

  std::cerr <<
       "General usage: you need to specify three prototext files, e.g:\n"
       "  srun -n# proto --model=<string> --optimizer=<string> --reader=<string>\n"
       "\n"
       "  However, if you are re-running an experiment from a previously saved\n"
       "  file, you only need to specify --model=<string>\n"
       "  When proto is run, an output file containing the concatenated prototext\n"
       "  files, along with other data is written. The default name for this file\n"
       "  is 'data.prototext'  You can specify an alternative name via the option:\n"
       "  --saveme=<string>  You can suppress writing the file via the option:\n"
       "  --saveme=0\n"
       "\n"
       "Some prototext values can be over-riden on the command line;\n"
       "(notes: use '1' or '0' for bool; if no value is given for a flag,\n"
       "        e.g: --use_cudnn, then a value of '1' is assigned)\n"
       "\n"
       "General:\n"
       "  --dag_model\n"
       "  --mini_batch_size=<int>\n"
       "  --num_epochs=<int>\n"
       "  --block_size=<int>\n"
       "  --procs_per_model=<int>\n"
       "  --num_gpus=<int>\n"
       "  --use_cudnn=<bool>\n"
       "     has no effect unless lbann was compiled with: __LIB_CUDNN\n"
       "  --random_seed=<int>\n"
       "  --objective_function<string>\n"
       "      <string> must be: categorical_cross_entropy or mean_squared_error\n"
       "  --data_layout<string>\n"
       "      <string> must be: data_parallel or model_parallel\n"
       "      note: this will be applied to all layers, metrics (and others)\n"
       "            that take DATA_PARALLEL or MODEL_PARALLEL as a template parameter\n"
       "  --print_affinity\n"
       "      display information on how OpenMP threads are provisioned\n"
       "\n"
       "DataReaders:\n"
       "  --data_filedir_train=<string>   --data_filedir_test=<string>\n"
       "  --data_filename_train=<string>  --data_filename_test=<string>\n"
       "  --label_filename_train=<string> --label_filename_test=<string>\n"
       "  --data_reader_percent=<float>\n"
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
       "\n"
       "  --opt=<string>\n"
       "     <string> must be one of:\n"
       "         adagrad, adam, hypergradient_adam, rmsprop, sgd\n"
       "\n"
       "  --learn_rate=< 0.01 >          (all except hypergradient_adam)\n"
       "  --eps=< 1e-8 >                 (all except sgd)\n"
       "  --beta1=< 0.9 >                (adam, hypergradient_adam)\n"
       "  --beta2=< 0.99 >               (adam, hypergradient_adam)\n"
       "  --init_learning_rate=< 0.01 >  (hypergradient_adam)\n"
       "  --hyper_learning_rate=< 1e-7 > (hypergradient_adam)\n"
       "  --momentum=< 0.9 >             (sgd)\n"
       "  --decay_rate=< 0.5 >           (sgd, rmsprop)\n"
       "  --nesterov=< false >           (sgd)\n";
}

void copy_file(std::string fn, std::ofstream &out)
{
  std::ifstream in(fn.c_str());
  if (!in.is_open()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for reading: " << fn;
    throw std::runtime_error(err.str());
  }
  std::stringstream s;
  s << in.rdbuf();
  out << s.str();
}

void save_session(lbann::lbann_comm *comm, int argc, char **argv, lbann_data::LbannPB& p)
{
  if (!comm->am_world_master()) {
    return;
  }

  options *opts = options::get();

  //do not write output file for a repeated experiment;
  //may want to revisit this decision later ...
  if (opts->has_string("loadme")) {
    return;
  }

  //get output filename
  std::string base = ".";
  if (!opts->has_string("saveme")) {
    std::cerr << "\nNOT WRITING SAVE_SESSION FILE since option --saveme=<string> is absent\n\n";
    return;
  }
  std::string name = opts->get_string("saveme");
  if (name == "0") {
    std::cerr << "\nNOT WRITING SAVE_SESSION FILE due to option: --saveme=0\n\n";
    return;
  }

  //check if "name" exists; if yes, append "_1"
  bool exists = false;
  std::ifstream in(name.c_str());
  if (in) {
    exists = true;
    in.close();
  }
  if (exists) {
    name += "_1";
    //opts["saveme"] = name;
  }

  //open output file
  std::ofstream out(name.c_str());
  if (!out.is_open()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for writing: " << name;
    throw std::runtime_error(err.str());
  }
  std::cout << std::endl << "writing options and prototext to file: " << name << "\n\n";

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
      << "#  $ srun -n" << comm->get_procs_in_world() << " " << argv[0]
      << " --loadme=" << opts->get_string("saveme") << "\n#\n#\n";

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
