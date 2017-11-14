#include "lbann/proto/proto_common.hpp"

#include "lbann/lbann.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unordered_map>
#include <sys/stat.h>
#include <memory> // for dynamic_pointer_cast

using namespace lbann;

/** Map from layer names to layers. */
std::map<std::string, Layer*> model_layers;

/** Whether a layer is already in the model. */
inline bool layer_is_in_model(std::string name) {
  return model_layers.find(name) != model_layers.end();
}

bool has_motifs(lbann_comm *comm, const lbann_data::LbannPB& p) {
  bool master = comm->am_world_master();
  if (master) cout << "starting has_motifs\n";
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
  if (master) cout << "starting expand_motifs\n";
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
  std::string name;
  std::stringstream ss;
  std::stringstream err;
  for (size_t i=0; i<proto_layers.size(); i++) {
    Layer *layer = model_layers[proto_layers[i].name()];

    // Set layer parents
    ss.clear();
    ss.str(proto_layers[i].parents());
    while (ss >> name) {
      if (master and not layer_is_in_model(name)) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not find parent layer " << name;
        throw lbann_exception(err.str());
      }
      Layer *parent_layer = model_layers[name];
      layer->add_parent_layer(parent_layer);
    }

    // Set layer children
    ss.clear();
    ss.str(proto_layers[i].children());
    while (ss >> name) {
      if (master and not layer_is_in_model(name)) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not find child layer " << name;
        throw lbann_exception(err.str());
      }
      Layer *child_layer = model_layers[name];
      layer->add_child_layer(child_layer);
    }

    // Set linked layers
    ss.clear();
    ss.str(proto_layers[i].linked_layers());
    while (ss >> name) {
      if (master and not layer_is_in_model(name)) {
        err << __FILE__ << " " << __LINE__ << " :: "
            << "could not find layer " << name << " to link with layer " << proto_layers[i].name();
        throw lbann_exception(err.str());
      }
      Layer *other_layer = model_layers[name];
      model->link_layers(other_layer, layer);
    }

    // Set a target layer's paired input layer
    if (dynamic_cast<target_layer*>(layer) != nullptr) {
      target_layer *target = dynamic_cast<target_layer*>(layer);

      // Get input layer name
      name.clear();
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
      input_layer *input = dynamic_cast<input_layer*>(model_layers[name]);
      target->set_paired_input_layer(input);

    }

    // Set a reconstruction layer's original layer
    if (proto_layers[i].has_reconstruction()) {

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

      // Set original layer
      Layer *original_layer = model_layers[name];
      if (dynamic_cast<reconstruction_layer<data_layout::MODEL_PARALLEL>*>(layer)) {
        reconstruction_layer<data_layout::MODEL_PARALLEL> *reconstruction
          = dynamic_cast<reconstruction_layer<data_layout::MODEL_PARALLEL>*>(layer);
        reconstruction->set_original_layer(original_layer);
      }
      if (dynamic_cast<reconstruction_layer<data_layout::DATA_PARALLEL>*>(layer)) {
        reconstruction_layer<data_layout::DATA_PARALLEL> *reconstruction
          = dynamic_cast<reconstruction_layer<data_layout::DATA_PARALLEL>*>(layer);
        reconstruction->set_original_layer(original_layer);
      }

    }

  }
}

lbann_callback_imcomm::comm_type get_comm_type(const string &s, bool master)
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

pool_mode get_pool_mode(const string& s, bool master)
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

weight_initialization get_weight_initialization(const string& s, bool master)
{
  if (s == "zero") {
    return weight_initialization::zero;
  } else if (s == "uniform") {
    return weight_initialization::uniform;
  } else if (s == "normal") {
    return weight_initialization::normal;
  } else if (s == "glorot_normal") {
    return weight_initialization::glorot_normal;
  } else if (s == "glorot_uniform") {
    return weight_initialization::glorot_uniform;
  } else if (s == "he_normal") {
    return weight_initialization::he_normal;
  } else if (s == "he_uniform") {
    return weight_initialization::he_uniform;
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " <<__LINE__
          << " :: unkown weight_initialization: " << s
          << " should be one of: zero uniform normal glorot_normal glorot_uniform he_normal he_uniform";
      throw lbann_exception(err.str());
    } else {
      return weight_initialization::zero;  //keep compiler happy, and have only one proc throw exception

    }
  }
}

inline data_layout get_data_layout(const string& s)
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

  const lbann_data::Model& m = p.model();
  std::vector<lbann_data::Layer> proto_layers;
  get_proto_layers(proto_layers, m, comm);

  Layer *d = nullptr;

  for (lbann_data::Layer& layer : proto_layers) {

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
        d = new relu_layer<data_layout::MODEL_PARALLEL>(comm, NULL);
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
          get_weight_initialization(ell.weight_initialization(), master),
          model->create_optimizer(),
          ell.has_bias(),
          ell.bias_initial_value(),
          cudnn);
      } else {
        d = new fully_connected_layer<data_layout::DATA_PARALLEL>(
          comm,
          num_neurons,
          get_weight_initialization(ell.weight_initialization(), master),
          model->create_optimizer(),
          ell.has_bias(),
          ell.bias_initial_value(),
          cudnn);
      }
      double l2_regularization_factor = ell.l2_regularization_factor();
      if(l2_regularization_factor != double(0.0)) {
        ((learning *) d)->set_l2_regularization_factor(l2_regularization_factor);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: reshape
    //////////////////////////////////////////////////////////////////
    else if (layer.has_reshape()) {
      const lbann_data::Reshape &ell = layer.reshape();
      int i;
      std::stringstream s(ell.dims());
      vector<int> dims;
      while (s >> i) {
        dims.push_back(i);
      }
      if (layout == data_layout::MODEL_PARALLEL) {
        d = new reshape_layer<data_layout::MODEL_PARALLEL>(comm, ell.num_dims(), dims.data());
      } else {
        d = new reshape_layer<data_layout::DATA_PARALLEL>(comm, ell.num_dims(), dims.data());
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
      vector<int> slice_points;
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

    //////////////////////////////////////////////////////////////////
    // LAYER: noise 
    //////////////////////////////////////////////////////////////////
    else if (layer.has_noise()) {
      const lbann_data::Noise& ell = layer.noise();
      d = new noise_layer<>(comm,ell.noise_factor(), cudnn);
    }

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
        vector<int> pool_dims;
        while (ss >> i) {
          pool_dims.push_back(i);
        }

        vector<int> pool_pads;
        ss.clear();
        ss.str(ell.pool_pads());
        while (ss >> i) {
          pool_pads.push_back(i);
        }

        vector<int> pool_strides;
        ss.clear();
        ss.str(ell.pool_strides());
        while (ss >> i) {
          pool_strides.push_back(i);
        }
        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: local_response_normalization "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new pooling_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            &pool_dims[0],
            &pool_pads[0],
            &pool_strides[0],
            get_pool_mode(ell.pool_mode(), master),
            cudnn
          );
        }
      } else {
        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: local_response_normalization "
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

    //////////////////////////////////////////////////////////////////
    // LAYER: unpooling
    //////////////////////////////////////////////////////////////////
    else if (layer.has_unpooling()) {
      const lbann_data::Unpooling& ell = layer.unpooling();
      pooling_layer<data_layout::DATA_PARALLEL> *pl = (pooling_layer<data_layout::DATA_PARALLEL>*)model_layers[ell.pooling_layer()];
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

    //////////////////////////////////////////////////////////////////
    // LAYER: Convolution
    //////////////////////////////////////////////////////////////////
    else if (layer.has_convolution()) {
      const lbann_data::Convolution& ell = layer.convolution();
      bool has_vectors = ell.has_vectors();

      if (has_vectors) {
        vector<int> conv_dims;
        std::stringstream ss;
        int i;
        ss.str(ell.conv_dims());
        while (ss >> i) {
          conv_dims.push_back(i);
        }

        vector<int> conv_pads;
        ss.clear();
        ss.str(ell.conv_pads());
        while (ss >> i) {
          conv_pads.push_back(i);
        }

        vector<int> conv_strides;
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
            &conv_dims[0],
            &conv_pads[0],
            &conv_strides[0],
            get_weight_initialization(ell.weight_initialization(), master),
            model->create_optimizer(),
            ell.has_bias(),
            ell.bias_initial_value(),
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
            get_weight_initialization(ell.weight_initialization(), master),
            model->create_optimizer(),
            ell.has_bias(),
            ell.bias_initial_value(),
            cudnn
          );
        }
      }

      double l2_regularization_factor = ell.l2_regularization_factor();
      if(l2_regularization_factor != double(0.0)) {
        ((learning *) d)->set_l2_regularization_factor(l2_regularization_factor);
      }
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: Deconvolution
    //////////////////////////////////////////////////////////////////
    else if (layer.has_deconvolution()) {
      const lbann_data::Deconvolution& ell = layer.deconvolution();
      bool has_vectors = ell.has_vectors();

      if (has_vectors) {
        vector<int> conv_dims;
        std::stringstream ss;
        int i;
        ss.str(ell.conv_dims());
        while (ss >> i) {
          conv_dims.push_back(i);
        }

        vector<int> conv_pads;
        ss.clear();
        ss.str(ell.conv_pads());
        while (ss >> i) {
          conv_pads.push_back(i);
        }

        vector<int> conv_strides;
        ss.clear();
        ss.str(ell.conv_strides());
        while (ss >> i) {
          conv_strides.push_back(i);
        }

        if (layout == data_layout::MODEL_PARALLEL and master) {
          err << __FILE__ << " " << __LINE__ << " :: deconvolution "
              << "does not support MODEL_PARALLEL layouts";
          throw lbann_exception(err.str());
        } else {
          d = new deconvolution_layer<data_layout::DATA_PARALLEL>(
            comm,
            ell.num_dims(),
            ell.num_output_channels(),
            &conv_dims[0],
            &conv_pads[0],
            &conv_strides[0],
            get_weight_initialization(ell.weight_initialization(), master),
            model->create_optimizer(),
            ell.has_bias(),
            ell.bias_initial_value(),
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
            get_weight_initialization(ell.weight_initialization(), master),
            model->create_optimizer(),
            ell.has_bias(),
            ell.bias_initial_value(),
            cudnn
          );
        }
      }

      double l2_regularization_factor = ell.l2_regularization_factor();
      if(l2_regularization_factor != double(0.0)) {
        ((learning *) d)->set_l2_regularization_factor(l2_regularization_factor);
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
          model->create_optimizer(),
          ell.decay(),
          ell.scale_init(),
          ell.bias_init(),
          ell.epsilon(),
          cudnn/*,
                 ell.global_stats()*/);
      }
    }

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
        d = new id_layer<data_layout::MODEL_PARALLEL>(comm);
      } else {
        d = new id_layer<data_layout::DATA_PARALLEL>(comm);
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
    }
    if (master and layer_is_in_model(d->get_name())) {
      err << __FILE__ << " " << __LINE__
          << " :: layer name " << layer_name << " is not unique" ;
      throw lbann_exception(err.str());
    }

    // Add layer to model
    model_layers[d->get_name()] = d;
    model->add(d);
    d = nullptr;

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
        cout << "constructing summarizer with dir: " << c.dir() << endl;
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


void get_layers_to_add_to_imcomm_callback(lbann_comm *comm, const lbann_data::Model& m, std::unordered_set<std::string> &addme, std::unordered_set<std::string> &excludeme) {
  bool master = comm->am_world_master();
  const int num_layers = m.layer_size();
  for (int j=0; j<num_layers; j++) {
    const lbann_data::Layer& layer = m.layer(j);
    switch (layer.imcomm()) {
      case lbann_data::Imcomm::DEFAULT :
        break;
      case lbann_data::Imcomm::EXCLUDE :
        excludeme.insert(layer.name());
        if (master) {
          std::cout << "EXPLICITLY EXCLUDING: " << layer.name() << std::endl;
        }
        break;
      case lbann_data::Imcomm::INCLUDE :
        addme.insert(layer.name());
        if (master) {
          std::cout << "EXPLICITLY INCLUDING: " << layer.name() << std::endl;
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
  if (master) cerr << endl << "starting init_callbacks; size: " << m.callback_size() << endl;

  
  //the same summarizer is passed to all call backs that take a summarizer;
  //construct_summarizer returns this summarizer, which may be a nullptr
  lbann_summary *summarizer = construct_summarizer(m, comm);


  //loop over the callbacks
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);

    //////////////////////////////////////////////////////////////////
    // CALLBACK: ltfb
    //////////////////////////////////////////////////////////////////
    if (callback.has_ltfb()) {
      const lbann_data::CallbackLTFB &c = callback.ltfb();
      lbann_callback_ltfb *ltfb_cb = new lbann_callback_ltfb(c.round_size(), summarizer);
      model->add_callback(ltfb_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: save_images
    //////////////////////////////////////////////////////////////////
    if (callback.has_save_images()) {
      const lbann_data::CallbackSaveImages& c = callback.save_images();
      string image_dir = c.image_dir();
      string extension = c.extension();
      generic_data_reader *reader = data_readers[execution_mode::training];
      lbann_callback_save_images *image_cb = new lbann_callback_save_images(reader, image_dir, extension);
      model->add_callback(image_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: print
    //////////////////////////////////////////////////////////////////
    if (callback.has_print()) {
      const lbann_data::CallbackPrint& c = callback.print();
      if (c.interval() > 0) {
        if (master) {
          cout << "adding print callback with interval: " << c.interval() << endl;
        }
        lbann_callback_print *print_cb = new lbann_callback_print(c.interval());
        model->add_callback(print_cb);
      }
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: timer
    //////////////////////////////////////////////////////////////////
    if (callback.has_timer()) {
      lbann_callback_timer *timer_cb = new lbann_callback_timer(summarizer);
      model->add_callback(timer_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: summary
    //////////////////////////////////////////////////////////////////
    if (callback.has_summary()) {
      const lbann_data::CallbackSummary& c = callback.summary();
      lbann_callback_summary *summary_cb = new lbann_callback_summary(summarizer, c.batch_interval(), c.mat_interval());
      model->add_callback(summary_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_weights
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_weights()) {
      const lbann_data::CallbackDumpWeights& c = callback.dump_weights();
      if (master) {
        cout << "adding dump weights callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
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
        cout << "adding dump activations callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
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
        cout << "adding dump gradients callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
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
        cout << "adding dump I/O callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
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
        cout << "adding callback to check the dataset" << endl;
      }
      lbann_callback_check_dataset *check_dataset_cb = new lbann_callback_check_dataset();
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
          cout << "adding display I/O stats callback for layer " << a;
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
        cout << "adding imcomm callback\n";
      }
      std::unordered_set<std::string> addme;
      std::unordered_set<std::string> excludeme;
      get_layers_to_add_to_imcomm_callback(comm, m, addme, excludeme);

      if (c.all_learning_layers()) {
        for (auto it : model_layers) {
          if (dynamic_cast<learning*>(it.second) != nullptr) {
            if (master) {
            }
            if (excludeme.find(it.second->get_name()) == excludeme.end()) {
              if (master) {
                std::cout << "ADDING to IMCOMM: " << it.second->get_name() 
                          << " " << it.second->get_type() << std::endl;
              } else {
                addme.insert(it.second->get_name());
              }  
            } else {
              if (master) {
                std::cout << "WOULD ADD TO IMCOMM, but was explicitly excluded: " 
                          << it.second->get_name() << " "
                          << it.second->get_type() << std::endl;
              } 
            }
          }
        }  
      }  
      std::unordered_set<Layer*> imcomm_layers;
      lbann_callback_imcomm::comm_type c_type  = get_comm_type(c.intermodel_comm_method(), master);
      lbann_callback_imcomm *im = new lbann_callback_imcomm(c_type, imcomm_layers, summarizer);
      model->add_callback(im);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: step_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_step_learning_rate()) {
      const lbann_data::CallbackStepLearningRate &c = callback.step_learning_rate();
      std::stringstream s(c.layers());
      std::unordered_set<Layer*> which;
      std::string a;
      bool all_layers = false;
      while (s >> a) {
        if (a == "10000") {
          all_layers = true;
        } else {
          if (master and not layer_is_in_model(a)) {
            err << __FILE__ << " " << __LINE__
                << " :: callback step_learning_rate: could not find layer " << a;
            throw lbann_exception(err.str());
          }
          which.insert(model_layers[a]);
        }
      }
      lbann_callback_adaptive_learning_rate *learn;
      if (all_layers) {
        learn = new lbann_callback_adaptive_learning_rate(c.step(), c.amt());
      } else {
        learn = new lbann_callback_adaptive_learning_rate(c.step(), c.amt(), which);
      }
      model->add_callback(learn);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: adaptive_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_adaptive_learning_rate()) {
      const lbann_data::CallbackAdaptiveLearningRate &c = callback.adaptive_learning_rate();
      std::stringstream s(c.layers());
      std::unordered_set<Layer*> which;
      string a;
      bool all_layers = false;
      while (s >> a) {
        if (a == "10000") {
          all_layers = true;
        } else {
          if (master and not layer_is_in_model(a)) {
            err << __FILE__ << " " << __LINE__
                << " :: callback adaptive_learning_rate: could not find layer " << a;
            throw lbann_exception(err.str());
          }
          which.insert(model_layers[a]);
        }
      }
      lbann_callback_adaptive_learning_rate *learn;
      if (all_layers) {
        learn = new lbann_callback_adaptive_learning_rate(c.patience(), c.amt());
      } else {
        learn = new lbann_callback_adaptive_learning_rate(c.patience(), c.amt(), which);
      }
      model->add_callback(learn);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: debug
    //////////////////////////////////////////////////////////////////
    if (callback.has_debug()) {
      const lbann_data::CallbackDebug& c = callback.debug();
      if (master) {
        cout << "adding debugging callback for phase: " << c.phase() << endl;
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
        cout << "adding debugging I/O callback for phase: " << c.phase() << endl;
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
      lbann_callback_checksmall *checksmall_cb = new lbann_callback_checksmall();
      model->add_callback(checksmall_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: check_nan
    //////////////////////////////////////////////////////////////////
    if (callback.has_check_nan()) {
      if (master) {
        std::cout << "adding check_nan callback" << std::endl;
      }
      lbann_callback_checknan *checknan_cb = new lbann_callback_checknan();
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
      lbann_callback_hang *hang_cb = new lbann_callback_hang(rank_to_hang);
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
      std::unordered_set<Layer*> layers;
      for (int i = 0; i < c.layer_size(); ++i) {
        layers.insert(model_layers[c.layer(i)]);
      }
      std::vector<int64_t> drop_epochs;
      for (int i = 0; i < c.drop_epoch_size(); ++i) {
        drop_epochs.push_back(c.drop_epoch(i));
      }
      lbann_callback_drop_fixed_learning_rate *dflr = new
      lbann_callback_drop_fixed_learning_rate(
        drop_epochs, c.amt(), layers);
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
      std::unordered_set<Layer*> layers;
      for (int i = 0; i < c.layer_size(); ++i) {
        layers.insert(model_layers[c.layer(i)]);
      }
      lbann_callback_linear_growth_learning_rate *lglr = new
      lbann_callback_linear_growth_learning_rate(
        c.target(), c.num_epochs(), c.delay(), layers);
      model->add_callback(lglr);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: profiler
    //////////////////////////////////////////////////////////////////
    if (callback.has_profiler()) {
      //const lbann_data::CallbackProfiler& c = callback.profiler();
      if (master) {
        cout << "adding profiler callback" << endl;
      }
      lbann_callback_profiler *profiler_cb = new lbann_callback_profiler();
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
      lbann_callback_step_minibatch *step_mb_cb = new
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

    //////////////////////////////////////////////////////////////////
    // CALLBACK: gradient_check
    //////////////////////////////////////////////////////////////////
    if (callback.has_gradient_check()) {
      const lbann_data::CallbackGradientCheck& c = callback.gradient_check();
      if (master) {
        std::cout << "adding gradient_check callback" << std::endl;
      }
      lbann_callback_gradient_check *gradient_check_cb = new
      lbann_callback_gradient_check(c.step_size(), c.verbose(), c.fail_on_error());
      model->add_callback(gradient_check_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: layerwise_adaptive_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_layerwise_adaptive_learning_rate()) {
      const lbann_data::CallbackLayerwiseAdaptiveLearningRate& c =
        callback.layerwise_adaptive_learning_rate();
      if (master) {
        std::cout << "adding layerwise_adaptive_learning_rate callback" <<
          " with scale=" << c.scale() << std::endl;
      }
      std::unordered_set<Layer*> layers;
      for (int i = 0; i < c.layer_size(); ++i) {
        layers.insert(model_layers[c.layer(i)]);
      }
      lbann_callback_layerwise_adaptive_learning_rate *lwalr_cb = new
        lbann_callback_layerwise_adaptive_learning_rate(c.scale(), layers);
      model->add_callback(lwalr_cb);
    }

  }

}


model *init_model(lbann_comm *comm, optimizer_factory *optimizer_fac, const lbann_data::LbannPB& p)
{
  std::stringstream err;
  bool master = comm->am_world_master();

  //sequential_model *model = 0;
  model *model = 0;

  const lbann_data::Model& m = p.model();
  const string name = m.name();
  const string obj_fn_name = m.objective_function();
  uint mini_batch_size = m.mini_batch_size();

  //instantiate the objective function
  objective_functions::objective_function *obj = 0;
  if (obj_fn_name == "cross_entropy") {
    obj = new objective_functions::cross_entropy();
  } else if (obj_fn_name == "mean_squared_error") {
    obj = new objective_functions::mean_squared_error();
  } else if (obj_fn_name == "binary_cross_entropy") {
    obj = new objective_functions::binary_cross_entropy();
  } else if (obj_fn_name == "geom_negloglike") {
    obj = new objective_functions::geom_negloglike();
  } else if (obj_fn_name == "mean_absolute_deviation") {
    obj = new objective_functions::mean_absolute_deviation();
  } else if (obj_fn_name == "poisson_negloglike") {
    obj = new objective_functions::poisson_negloglike();
  } else {
    if (master) {
      err << __FILE__ << " " << __LINE__
          << " :: init_model() - unknown objective function name: " << obj_fn_name
          << std::endl << "; should be one of: cross_entropy, mean_squared_error";
      throw lbann_exception(err.str());
    }
  }

  //instantiate the network; layers will be added in a separate function call
  if (name == "sequential_model" || name == "dnn") {
    if (master && name == "dnn") std::cout << "WARNING: \"dnn\" model is deprecated in favor of \"sequential_model\"\n";
    model = new sequential_model(mini_batch_size, comm, obj, optimizer_fac);
    if (master) std::cout << "instantiating sequential_model\n";
  } else if (name == "dag_model") {
    model = new dag_model(mini_batch_size, comm, obj, optimizer_fac);
    if (master) std::cout << "instantiating dag_model\n";
  } else if(name == "planar_model") {
/// XXX
/// Settting the number of heads to 3 temporarly; will be fixed as a parameter
    model = new planar_model(mini_batch_size, comm, obj, optimizer_fac, 3);
    if (master) std::cout << "instantiating planar_model\n";
  } else if (name == "greedy_layerwise_autoencoder") {
    model = new greedy_layerwise_autoencoder(mini_batch_size, comm, obj, optimizer_fac);
    if (master) std::cout << "instantiating greedy_layerwise_autoencoder\n";
  } else {
    if (master) {
      err << __FILE__ << " " << __LINE__
          << " :: init_model() - unknown model name: " << name << endl
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
      if (layout == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::categorical_accuracy<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::categorical_accuracy<data_layout::DATA_PARALLEL>(comm));
      }
    } else if (metric.has_mean_squared_error()) {
      if (layout == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::mean_squared_error<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::mean_squared_error<data_layout::DATA_PARALLEL>(comm));
      }
    } else if (metric.has_pearson_correlation()) {
      if (layout == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::pearson_correlation<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::pearson_correlation<data_layout::DATA_PARALLEL>(comm));
      }
    } else if (metric.has_top_k_categorical_accuracy()) {
      const lbann_data::TopKCategoricalAccuracy &a = metric.top_k_categorical_accuracy();
      if (layout == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::top_k_categorical_accuracy<data_layout::MODEL_PARALLEL>(a.top_k(), comm));
      } else {
        model->add_metric(new metrics::top_k_categorical_accuracy<data_layout::DATA_PARALLEL>(a.top_k(), comm));
      }
    }
  }

  //set checkpoint values
  model->set_checkpoint_dir(m.checkpoint_dir());
  model->set_checkpoint_epochs(m.checkpoint_epochs());
  model->set_checkpoint_steps(m.checkpoint_steps());
  model->set_checkpoint_secs(m.checkpoint_secs());

  return model;
}

optimizer_factory *init_optimizer_factory(lbann_comm *comm, cudnn::cudnn_manager *cudnn,
    const lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();
  optimizer_factory *factory = 0;
  const lbann_data::Optimizer &opt = p.optimizer();
  if (opt.has_adagrad()) {
    const lbann_data::Adagrad &a = opt.adagrad();
    factory = new adagrad_factory(comm, a.learn_rate(), a.eps());
  } else if (opt.has_rmsprop()) {
    const lbann_data::Rmsprop &a = opt.rmsprop();
    factory = new rmsprop_factory(comm, a.learn_rate(), a.decay_rate(), a.eps());
  } else if (opt.has_adam()) {
    const lbann_data::Adam &a = opt.adam();
    factory = new adam_factory(comm, a.learn_rate(), a.beta1(), a.beta2(), a.eps(), cudnn);
  } else if (opt.has_hypergradient_adam()) {
    const lbann_data::HypergradientAdam &a = opt.hypergradient_adam();
    factory = new hypergradient_adam_factory(comm, a.init_learning_rate(), a.hyper_learning_rate(), a.beta1(), a.beta2(), a.eps());
  } else if (opt.has_sgd()) {
    const lbann_data::Sgd &a = opt.sgd();
    factory = new sgd_factory(comm, a.learn_rate(), a.momentum(), a.decay_rate(), a.nesterov());
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: init_optimizer_factory: prototext does not appear to contain an optimizer!";
      throw lbann_exception(err.str());
    }
  }

  return factory;
}


void init_image_preprocessor(const lbann_data::Reader& pb_readme, const bool master,
                             std::shared_ptr<cv_process>& pp, int& width, int& height) {
  if (!pb_readme.has_image_preprocessor()) return;

  const lbann_data::ImagePreprocessor& pb_preprocessor = pb_readme.image_preprocessor();
  if (pb_preprocessor.disable()) return;

  // data reader name
  const string& name = pb_readme.name();
  // final size of image
  width = pb_preprocessor.raw_width();
  height = pb_preprocessor.raw_height();

  // set up a cropper
  if (pb_preprocessor.has_cropper()) {
    const lbann_data::ImagePreprocessor::Cropper& pb_cropper = pb_preprocessor.cropper();
    if (!pb_cropper.disable()) {
      const string cropper_name = ((pb_cropper.name() == "")? "default_cropper" : pb_cropper.name());
      std::unique_ptr<lbann::cv_cropper> cropper(new(lbann::cv_cropper));
      cropper->set_name(cropper_name);
      cropper->set(pb_cropper.crop_width(),
                   pb_cropper.crop_height(),
                   pb_cropper.crop_randomly(),
                   std::make_pair<int,int>(pb_cropper.resized_width(),
                                           pb_cropper.resized_height()));
      pp->add_transform(std::move(cropper));
      width = pb_cropper.crop_width();
      height = pb_cropper.crop_height();
      if (master) cout << "image processor: " << cropper_name << " cropper is set" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    if(pb_preprocessor.crop_first()) {
      std::unique_ptr<lbann::cv_cropper> cropper(new(lbann::cv_cropper));
      cropper->set(pb_preprocessor.crop_width(),
                   pb_preprocessor.crop_height(),
                   pb_preprocessor.crop_randomly(),
                   std::make_pair<int,int>(pb_preprocessor.resized_width(),
                                           pb_preprocessor.resized_height()));
      pp->add_transform(std::move(cropper));
      if (master) cout << "image processor: cropper is set (deprecated syntax)" << endl;
    }
  }

  // set up an augmenter
  if (pb_preprocessor.has_augmenter()) {
    const lbann_data::ImagePreprocessor::Augmenter& pb_augmenter = pb_preprocessor.augmenter();
    if (!pb_augmenter.disable() &&
        (pb_augmenter.horizontal_flip() ||
         pb_augmenter.vertical_flip() ||
         pb_augmenter.rotation() != 0.0 ||
         pb_augmenter.horizontal_shift() != 0.0 ||
         pb_augmenter.vertical_shift() != 0.0 ||
         pb_augmenter.shear_range() != 0.0))
    {
      const string augmenter_name = ((pb_augmenter.name() == "")? "default_augmenter" : pb_augmenter.name());
      std::unique_ptr<lbann::cv_augmenter> augmenter(new(lbann::cv_augmenter));
      augmenter->set_name(augmenter_name);
      augmenter->set(pb_augmenter.horizontal_flip(),
                     pb_augmenter.vertical_flip(),
                     pb_augmenter.rotation(),
                     pb_augmenter.horizontal_shift(),
                     pb_augmenter.vertical_shift(),
                     pb_augmenter.shear_range());
      pp->add_transform(std::move(augmenter));
      if (master) cout << "image processor: " << augmenter_name << " augmenter is set" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    if (!pb_preprocessor.disable_augmentation()) {
      std::unique_ptr<lbann::cv_augmenter> augmenter(new(lbann::cv_augmenter));
      augmenter->set(pb_preprocessor.horizontal_flip(),
                   pb_preprocessor.vertical_flip(),
                   pb_preprocessor.rotation(),
                   pb_preprocessor.horizontal_shift(),
                   pb_preprocessor.vertical_shift(),
                   pb_preprocessor.shear_range());
      pp->add_transform(std::move(augmenter));
      if (master) cout << "image processor: augmenter is set (deprecated syntax)" << endl;
    }
  }

  // set up a colorizer
  if (pb_preprocessor.has_colorizer()) {
    const lbann_data::ImagePreprocessor::Colorizer& pb_colorizer = pb_preprocessor.colorizer();
    if  (!pb_colorizer.disable()) {
      const string colorizer_name = ((pb_colorizer.name() == "")? "default_colorizer" : pb_colorizer.name());
      // If every image in the dataset is a color image, this is not needed
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      colorizer->set_name(colorizer_name);
      pp->add_transform(std::move(colorizer));
      if (master) cout << "image processor: " << colorizer_name << " colorizer is set" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    if (!pb_preprocessor.no_colorize()) {
      std::unique_ptr<lbann::cv_colorizer> colorizer(new(lbann::cv_colorizer));
      pp->add_transform(std::move(colorizer));
      if (master) cout << "image processor: colorizer is set (deprecated syntax)" << endl;
    }
  }

  // set up a normalizer
  if (pb_preprocessor.has_normalizer()) {
    const lbann_data::ImagePreprocessor::Normalizer& pb_normalizer = pb_preprocessor.normalizer();
    if (!pb_normalizer.disable()) {
      const string normalizer_name = ((pb_normalizer.name() == "")? "default_normalizer" : pb_normalizer.name());
      std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
      normalizer->set_name(normalizer_name);
      normalizer->unit_scale(pb_normalizer.scale());
      normalizer->subtract_mean(pb_normalizer.subtract_mean());
      normalizer->unit_variance(pb_normalizer.unit_variance());
      normalizer->z_score(pb_normalizer.z_score());
      pp->add_normalizer(std::move(normalizer));
      if (master) cout << "image processor: " << normalizer_name << " normalizer is set" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    std::unique_ptr<lbann::cv_normalizer> normalizer(new(lbann::cv_normalizer));
    normalizer->unit_scale(pb_preprocessor.scale());
    normalizer->subtract_mean(pb_preprocessor.subtract_mean());
    normalizer->unit_variance(pb_preprocessor.unit_variance());
    normalizer->z_score(pb_preprocessor.z_score());
    pp->add_normalizer(std::move(normalizer));
    if (master) cout << "image processor: normalizer is set (deprecated syntax)" << endl;
  }

  // set up a noiser
  if (pb_preprocessor.has_noiser()) {
    const lbann_data::ImagePreprocessor::Noiser& pb_noiser = pb_preprocessor.noiser();
    if (!pb_noiser.disable()) {
      const string noiser_name = ((pb_noiser.name() == "")? "default_noiser" : pb_noiser.name());
/* TODO: implement noiser in opencv
      std::unique_ptr<lbann::cv_noiser> noiser(new(lbann::cv_noiser));
      noiser->set_name(noiser_name);
      noiser->set(pb_noiser.factor());
      pp->add_transform(std::move(noiser));
*/
      if (master) cout << "image processor: " << noiser_name << " noiser is not supported yet" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
/* TODO: implement noiser in opencv
    std::unique_ptr<lbann::cv_noiser> noiser(new(lbann::cv_noiser));
    noiser->set(pb_preprocessor.noise_factor());
    pp->add_transform(std::move(noiser));
*/
    if (master && (pb_preprocessor.noise_factor() > 0.0))
        cout << "image processor: noiser is not supported yet (deprecated syntax)" << endl;
  }

  // create a data reader
  if (name == "imagenet_patches") {
    std::shared_ptr<cv_process_patches> ppp = std::dynamic_pointer_cast<cv_process_patches>(pp);
    if (pb_preprocessor.has_patch_extractor()) {
      const lbann_data::ImagePreprocessor::PatchExtractor& pb_patch_extractor = pb_preprocessor.patch_extractor();
      if (!pb_patch_extractor.disable()) {
        const string patch_extractor_name = ((pb_patch_extractor.name() == "")? "default_patch_extractor" : pb_patch_extractor.name());
        lbann::patchworks::patch_descriptor pi;
        pi.set_sample_image(static_cast<unsigned int>(width),
                            static_cast<unsigned int>(height));
        pi.set_size(pb_patch_extractor.patch_width(), pb_patch_extractor.patch_height ());
        pi.set_gap(pb_patch_extractor.patch_gap());
        pi.set_jitter(pb_patch_extractor.patch_jitter());
        pi.set_mode_centering(pb_patch_extractor.centering_mode());
        pi.set_mode_chromatic_aberration(pb_patch_extractor.ca_correction_mode());
        pi.set_self_label();
        pi.define_patch_set();
        ppp->set_name(patch_extractor_name);
        ppp->set_patch_descriptor(pi);
        if (master) cout << "image processor: " << patch_extractor_name << " patch_extractor is set" << endl;
      }
    }
  }
}


void init_image_data_reader(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* &reader) {
  // data reader name
  const string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();
  // number of labels
  const int n_labels = pb_readme.num_labels();

  std::shared_ptr<cv_process> pp;
  // set up the image preprocessor
  if ((name == "imagenet") || (name == "imagenet_single")) {
    pp = std::make_shared<cv_process>();
  } else if (name == "imagenet_patches") {
    pp = std::make_shared<cv_process_patches>();
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: unknown name for image data reader: "
          << name;
      throw lbann_exception(err.str());
    }
  }

  // final size of image
  int width = 0, height = 0;

  // setup preprocessor
  init_image_preprocessor(pb_readme, master, pp, width, height);

  if (name == "imagenet_patches") {
    std::shared_ptr<cv_process_patches> ppp = std::dynamic_pointer_cast<cv_process_patches>(pp);
    reader = new imagenet_reader_patches(ppp, shuffle);
    if (master) cout << "imagenet_reader_patches is set" << endl;
  } else if (name == "imagenet") {
    reader = new imagenet_reader(pp, shuffle);
    if (master) cout << "imagenet_reader is set" << endl;
  } else { // imagenet_single
    reader = new imagenet_reader_single(pp, shuffle);
    if (master) cout << "imagenet_reader_single is set" << endl;
  }

  image_data_reader* image_data_reader_ptr = dynamic_cast<image_data_reader*>(reader);
  if (!image_data_reader_ptr && master) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: invalid image data reade pointer";
    throw lbann_exception(err.str());
  }

  // configure the data reader
  image_data_reader_ptr->set_input_params(width, height, 3, n_labels);
}


void init_generic_preprocessor(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* reader) {
  if (!pb_readme.has_image_preprocessor()) return;

  const lbann_data::ImagePreprocessor& pb_preprocessor = pb_readme.image_preprocessor();
  if (pb_preprocessor.disable()) return;

  // set up augmenter if necessary
  if (pb_preprocessor.has_augmenter()) {
    const lbann_data::ImagePreprocessor::Augmenter& pb_augmenter = pb_preprocessor.augmenter();
    if (!pb_augmenter.disable() &&
        (pb_augmenter.name() == "") &&
        (pb_augmenter.horizontal_flip() ||
         pb_augmenter.vertical_flip() ||
         pb_augmenter.rotation() != 0.0 ||
         pb_augmenter.horizontal_shift() != 0.0 ||
         pb_augmenter.vertical_shift() != 0.0 ||
         pb_augmenter.shear_range() != 0.0))
    {
      reader->horizontal_flip( pb_augmenter.horizontal_flip() );
      reader->vertical_flip( pb_augmenter.vertical_flip() );
      reader->rotation( pb_augmenter.rotation() );
      reader->horizontal_shift( pb_augmenter.horizontal_shift() );
      reader->vertical_shift( pb_augmenter.vertical_shift() );
      reader->shear_range( pb_augmenter.shear_range() );
      if (master) cout << "image processor: augmenter is set" << endl;
    } else {
      reader->disable_augmentation();
    }
  } else { // For backward compatibility. TODO: will be deprecated
    if (!pb_preprocessor.disable_augmentation() &&
        (pb_preprocessor.horizontal_flip() ||
         pb_preprocessor.vertical_flip() ||
         pb_preprocessor.rotation() != 0.0 ||
         pb_preprocessor.horizontal_shift() != 0.0 ||
         pb_preprocessor.vertical_shift() != 0.0 ||
         pb_preprocessor.shear_range() != 0.0)) {
      reader->horizontal_flip( pb_preprocessor.horizontal_flip() );
      reader->vertical_flip( pb_preprocessor.vertical_flip() );
      reader->rotation( pb_preprocessor.rotation() );
      reader->horizontal_shift( pb_preprocessor.horizontal_shift() );
      reader->vertical_shift( pb_preprocessor.vertical_shift() );
      reader->shear_range( pb_preprocessor.shear_range() );
      if (master) cout << "image processor: deprecated syntax for augmenter" << endl;
    }
  }

  // set up the normalizer
  if (pb_preprocessor.has_normalizer()) {
    const lbann_data::ImagePreprocessor::Normalizer& pb_normalizer = pb_preprocessor.normalizer();
    if (!pb_normalizer.disable() &&
        (pb_normalizer.name() == "")) {
      reader->subtract_mean( pb_normalizer.subtract_mean() );
      reader->unit_variance( pb_normalizer.unit_variance() );
      reader->scale( pb_normalizer.scale() );
      reader->z_score( pb_normalizer.z_score() );
      if (master) cout << "image processor: normalizer is set" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
      reader->subtract_mean( pb_preprocessor.subtract_mean() );
      reader->unit_variance( pb_preprocessor.unit_variance() );
      reader->scale( pb_preprocessor.scale() );
      reader->z_score( pb_preprocessor.z_score() );
      if (master) cout << "image processor: deprecated syntax for normalizer" << endl;
  }

  if (pb_preprocessor.has_noiser()) {
    const lbann_data::ImagePreprocessor::Noiser& pb_noiser = pb_preprocessor.noiser();
    if (!pb_noiser.disable() &&
        (pb_noiser.name() == "")) {
      reader->add_noise( pb_noiser.factor() );
      if (master) cout << "image processor: noiser is set" << endl;
    }
  } else { // For backward compatibility. TODO: will be deprecated
    reader->add_noise( pb_preprocessor.noise_factor() );
    if (master && (pb_preprocessor.noise_factor()>0.0)) cout << "image processor: deprecated syntax for noiser" << endl;
  }
}


void init_org_image_data_reader(const lbann_data::Reader& pb_readme, const bool master, generic_data_reader* &reader) {
  const lbann_data::ImagePreprocessor& pb_preprocessor = pb_readme.image_preprocessor();

  // data reader name
  const string& name = pb_readme.name();
  // whether to shuffle data
  const bool shuffle = pb_readme.shuffle();
  // final size of image. If image_preprocessor is not set, the type-default value
  // (i,e., 0) is used. Then,set_input_params() will not modify the current member value.
  const int width = pb_preprocessor.raw_width();
  const int height = pb_preprocessor.raw_height();

  // number of labels
  const int n_labels = pb_readme.num_labels();

  // TODO: as imagenet_org phases out, and mnist and cifar10 convert to use new
  // imagenet data reader, this function will disappear
  // create data reader
  if (name == "imagenet_org") {
    reader = new imagenet_reader_org(shuffle);
    dynamic_cast<imagenet_reader_org*>(reader)->set_input_params(width, height, 3, n_labels);
    if (master) cout << "imagenet_reader_org is set" << endl;
  } else if (name == "mnist") {
    reader = new mnist_reader(shuffle);
    if (master) cout << "mnist_reader is set" << endl;
  } else if (name == "cifar10") {
    reader = new cifar10_reader(shuffle);
    if (master) cout << "cifar10_reader is set" << endl;
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: unknown name for image data reader: "
          << name;
      throw lbann_exception(err.str());
    }
  }

  // setup preprocessor
  init_generic_preprocessor(pb_readme, master, reader);
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

    const string& name = readme.name();

    const bool shuffle = readme.shuffle();

    generic_data_reader *reader = 0;
    generic_data_reader *reader_validation = 0;

    if ((name == "imagenet_org") || (name == "mnist") || (name == "cifar10")) {
      init_org_image_data_reader(readme, master, reader);
      set_up_generic_preprocessor = false;
    } else if ((name == "imagenet") || (name == "imagenet_single") || (name == "imagenet_patches")) {
      init_image_data_reader(readme, master, reader);
      set_up_generic_preprocessor = false;
    } else if (name == "nci") {
      reader = new data_reader_nci(shuffle);
    } else if (name == "csv") {
      csv_reader* reader_csv = new csv_reader(shuffle);
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
      numpy_reader* reader_numpy = new numpy_reader(shuffle);
      reader_numpy->set_has_labels(!readme.disable_labels());
      reader_numpy->set_has_responses(!readme.disable_responses());
      reader = reader_numpy;
    } else if (name == "merge_samples") {
      auto paths = glob(readme.data_file_pattern());
      std::vector<generic_data_reader*> npy_readers;
      for (const auto path : paths) {
        if (readme.format() == "numpy") {
          numpy_reader *reader_numpy = new numpy_reader(false);
          reader_numpy->set_data_filename(path);
          reader_numpy->set_has_labels(!readme.disable_labels());
          reader_numpy->set_has_responses(!readme.disable_responses());
          npy_readers.push_back(reader_numpy);
        } else if (readme.format() == "csv") {
          csv_reader* reader_csv = new csv_reader(shuffle);
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
      data_reader_merge_samples* merged_reader = new data_reader_merge_samples(npy_readers, shuffle);
      reader = merged_reader;
    } else if (name == "synthetic") {
      reader = new data_reader_synthetic(readme.num_samples(), readme.num_features(), shuffle);
    } else if (name == "ascii") {
      reader = new ascii_reader(5, shuffle);
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
    if (readme.label_filename() != "") {
      reader->set_label_filename( readme.label_filename() );
    }
    if (readme.data_filedir() != "") {
      reader->set_file_dir( readme.data_filedir() );
    }

    reader->set_absolute_sample_count( readme.absolute_sample_count() );
    reader->set_use_percent( readme.percent_of_data_to_use() );

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
      } else if (name == "cifar10") {
        reader_validation = new cifar10_reader(shuffle);
        (*(cifar10_reader *)reader_validation) = (*(cifar10_reader *)reader);
        /*
        } else if (name == "synthetic") {
        reader_validation = new data_reader_synthetic(shuffle);
        */
      } else if (name == "ascii") {
        reader_validation = new ascii_reader(5, shuffle);
        (*(ascii_reader *)reader_validation) = (*(ascii_reader *)reader);
      }

      reader_validation->swap_role("validate");
      reader_validation->use_unused_index_set();

      if (master) {
        size_t num_train = reader->get_num_data();
        size_t num_validate = reader_validation->get_num_data();
        double validate_percent = ((double) num_validate / (double) (num_train+num_validate))*100.0;
        double train_percent = ((double) num_train / (double) (num_train+num_validate))*100.0;
        cout << "Training using " << train_percent << "% of the training data set, which is " << reader->get_num_data() << " samples." << endl
             << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->get_num_data() << " samples." << endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }
}

void read_prototext_file(string fn, lbann_data::LbannPB& pb, bool master)
{
  std::stringstream err;
  int fd = open(fn.c_str(), O_RDONLY);
  if (fd == -1) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
      throw lbann_exception(err.str());
    }
  }
  google::protobuf::io::FileInputStream *input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, &pb);
  if (!success) {
    if (master) {
      err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << endl;
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
  google::protobuf::io::FileOutputStream *output = new google::protobuf::io::FileOutputStream(fd);
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
      cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
           " (Limited to # Processes)" << endl;
    }
    parallel_io = comm->get_procs_per_model();
    model->set_num_parallel_readers(parallel_io); //adjust the prototext
  } else {
    if (master) {
      cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
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

    lbann_data::Optimizer *opt = new lbann_data::Optimizer;

    //construct the new optimizer
    std::string opt_string = opts->get_string("opt");
    if (opt_string == "adagrad") {
      lbann_data::Adagrad *a = new lbann_data::Adagrad;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      opt->set_allocated_adagrad(a);
    } else if (opt_string == "adam") {
      lbann_data::Adam *a = new lbann_data::Adam;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      opt->set_allocated_adam(a);
    } else if (opt_string == "hypergradient_adam") {
      lbann_data::HypergradientAdam *a = new lbann_data::HypergradientAdam;
      a->set_init_learning_rate(init_learning_rate);
      a->set_hyper_learning_rate(hyper_learning_rate);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      a->set_eps(eps);
      opt->set_allocated_hypergradient_adam(a);
    } else if (opt_string == "rmsprop") {
      lbann_data::Rmsprop *a = new lbann_data::Rmsprop;
      a->set_learn_rate(learn_rate);
      a->set_decay_rate(decay_rate);
      a->set_eps(eps);
      opt->set_allocated_rmsprop(a);
    } else if (opt_string == "sgd") {
      if (master) std::cerr << "\n\nsetting: sgd\n\n";
      lbann_data::Sgd *a = new lbann_data::Sgd;
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

  cout << endl
       << "Running with these parameters:\n"
       << " General:\n"
       << "  datatype size:        " << sizeof(DataType) << endl
       << "  mini_batch_size:      " << m.mini_batch_size() << endl
       << "  num_epochs:           " << m.num_epochs()  << endl
       << "  block_size:           " << m.block_size()  << endl
       << "  procs_per_model:      " << m.procs_per_model()  << endl
       << "  num_gpus:             " << m.num_gpus()  << endl
       << "  num_parallel_readers: " << m.num_parallel_readers()  << endl
       << "  use_cudnn:            " << m.use_cudnn()  << endl
       << "  random_seed:          " << m.random_seed() << endl
       << "  objective_function:   " << m.objective_function()  << endl
       << "  data_layout:          " << m.data_layout()  << endl
       << "     (only used for metrics)\n"
       << "\n"
       << " Optimizer:  ";

  const lbann_data::Optimizer &o = p.optimizer();
  if (o.has_adagrad()) {
    const lbann_data::Adagrad &a = o.adagrad();
    cout << "  Adagrad\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  eps:        " << a.eps()  << endl;
  } else if (o.has_rmsprop()) {
    const lbann_data::Rmsprop &a = o.rmsprop();
    cout <<  "  Rmsprop\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  decay_rate: " << a.decay_rate()  << endl
         << "  eps:        " << a.eps()  << endl;
  } else if (o.has_adam()) {
    const lbann_data::Adam &a = o.adam();
    cout << "  Adam\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  beta1:      " << a.beta1()  << endl
         << "  beta2:      " << a.beta2()  << endl
         << "  eps:        " << a.eps()  << endl;
  } else if (o.has_hypergradient_adam()) {
    const lbann_data::HypergradientAdam &a = o.hypergradient_adam();
    cout << "  HypergradientAdam\n"
         << "  init_learning_rate:  " << a.init_learning_rate()  << endl
         << "  hyper_learning_rate: " << a.hyper_learning_rate()  << endl
         << "  beta1:               " << a.beta1()  << endl
         << "  beta2:               " << a.beta2()  << endl
         << "  eps:                 " << a.eps()  << endl;
  } else if (o.has_sgd()) {
    const lbann_data::Sgd &a = o.sgd();
    cout << "  Sgd\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  momentum:   " << a.momentum()  << endl
         << "  decay_rate: " << a.decay_rate()  << endl
         << "  nesterov:   " << a.nesterov()  << endl;
  }
}

void print_help(lbann::lbann_comm *comm)
{
  if (!comm->am_world_master()) {
    return;
  }

  cerr <<
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
       "\n"
       "DataReaders:\n"
       "  --data_filedir_train=<string>   --data_filedir_test=<string>\n"
       "  --data_filename_train=<string>  --data_filename_test=<string>\n"
       "  --label_filename_train=<string> --label_filename_test=<string>\n"
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
  ifstream in(name.c_str());
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

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::time_t r = std::time(nullptr);
  char *tm = std::ctime(&r);
  size_t fixme = strlen(tm);
  tm[fixme-1] = 0;
  out << "\n#\n# Experiment conducted at: "
      <<  tm
      << "\n#\n#\n# Experiment was run with lbann version: "
      << lbann_version << "\n#\n#\n# To rerun the experiment: \n"
      << "#  $ srun -n" << size << " " << argv[0]
      << " --loadme=" << opts->get_string("saveme") << "\n#\n#\n";

  out << "# Selected SLURM Environment Variables:\n";
  std::vector<std::string> v = {"HOST", "SLURM_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE"};
  for (size_t i=0; i<v.size(); i++) {
    char *c = std::getenv(v[i].c_str());
    if (c != 0) {
      out << "# " << v[i] << "=" << c << std::endl;
    }
  }
  out << "\n#\n#\n";

  std::string s;
  google::protobuf::TextFormat::PrintToString(p, &s);
  out << s;
  out.close();
}
