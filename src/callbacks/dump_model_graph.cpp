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

#include "lbann/callbacks/dump_model_graph.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann_config.hpp"

#include "lbann/proto/callbacks.pb.h"

#ifdef LBANN_HAS_BOOST
#include <boost/graph/graphviz.hpp>
#endif // LBANN_HAS_BOOST

namespace lbann {
namespace callback {

#ifdef LBANN_HAS_BOOST
namespace {
struct vertex_property
{
  std::string label;
  std::string color;
  std::string shape;
  std::string style;
};
typedef boost::property<boost::edge_name_t, std::string> edge_property;
typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::directedS,
                              vertex_property,
                              edge_property>
  model_graph;
} // namespace
#endif // LBANN_HAS_BOOST

void dump_model_graph::on_setup_end(model* m)
{
#ifdef LBANN_HAS_BOOST
  const auto layers = m->get_layers();
  if (!m->get_comm()->am_world_master())
    return;

  const auto get_tensor_dims_str = [](const std::vector<int>& dims) {
    std::stringstream ss;
    for (unsigned int i = 0; i < dims.size(); i++)
      ss << (i ? "x" : "") << dims[i];
    return ss.str();
  };

  if (m_print) {
    std::stringstream ss;
    ss << "dump_model_graph callback:";
    for (const Layer* layer : layers) {
      ss << "   " << layer->get_name() << ": ";
      if (layer->get_num_parents()) {
        for (int i = 0; i < layer->get_num_parents(); i++) {
          const std::vector<int> input_dims = layer->get_input_dims(i);
          ss << (i ? ", " : "") << get_tensor_dims_str(input_dims) << " ("
             << layer->get_parent_layer(i).get_name() << ")";
        }
      }
      else {
        ss << "no input";
      }
      ss << " -> ";
      if (layer->get_num_children()) {
        for (int i = 0; i < layer->get_num_children(); i++) {
          const std::vector<int> output_dims = layer->get_output_dims(i);
          ss << (i ? ", " : "") << get_tensor_dims_str(output_dims) << " ("
             << layer->get_child_layer(i).get_name() << ")";
        }
      }
      else {
        ss << "no output";
      }
      ss << std::endl;
    }
    std::cout << ss.str();
  }

  const auto get_vertex_color = [](const std::string layer_type) {
    const size_t hash =
      (std::hash<std::string>()(layer_type) & 0xFFFFFF) | 0x808080;
    std::stringstream ssh;
    ssh << std::hex << hash;
    const std::string hex = ssh.str();
    std::stringstream ss;
    ss << "#" << std::string(6 - hex.size(), '0') << hex;
    return ss.str();
  };

  model_graph g;
  std::vector<model_graph::vertex_descriptor> vertexs;
  for (unsigned int i = 0; i < layers.size(); i++)
    vertexs.push_back(boost::add_vertex(g));

  std::set<std::tuple<const Layer*, const Layer*, std::string>> edges;
  for (unsigned int i = 0; i < layers.size(); i++) {
    const auto l = layers[i];
    const auto v = vertexs[i];
    g[v].label = l->get_name() + "\\n(" + l->get_type() + ")";
    g[v].color = get_vertex_color(l->get_type());
    g[v].shape = l->get_num_parents() == 0 || l->get_num_children() == 0
                   ? "ellipse"
                   : "box";
    g[v].style = "filled";
    for (int i_parent = 0; i_parent < l->get_num_parents(); i_parent++)
      edges.emplace(l->get_parent_layers()[i_parent],
                    l,
                    get_tensor_dims_str(l->get_input_dims(i_parent)));
    for (int i_child = 0; i_child < l->get_num_children(); i_child++)
      edges.emplace(l,
                    l->get_child_layers()[i_child],
                    get_tensor_dims_str(l->get_output_dims(i_child)));
  }
  for (const auto& edge : edges) {
    const auto i_from =
      std::distance(layers.begin(),
                    std::find(layers.begin(), layers.end(), std::get<0>(edge)));
    const auto i_to =
      std::distance(layers.begin(),
                    std::find(layers.begin(), layers.end(), std::get<1>(edge)));
    boost::add_edge(vertexs[i_from],
                    vertexs[i_to],
                    edge_property(std::get<2>(edge)),
                    g);
  }

  boost::dynamic_properties dp;
  dp.property("label", get(&vertex_property::label, g));
  dp.property("color", get(&vertex_property::color, g));
  dp.property("shape", get(&vertex_property::shape, g));
  dp.property("style", get(&vertex_property::style, g));
  dp.property("node_id", get(boost::vertex_index, g));
  dp.property("label", get(boost::edge_name, g));
  std::ofstream file(m_basename);
  boost::write_graphviz_dp(file, g, dp);

#else  // LBANN_HAS_BOOST
  (void)m_print;
  LBANN_ERROR("This callback requires the Boost library.");
#endif // LBANN_HAS_BOOST
}

void dump_model_graph::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_dump_model_graph();
  msg->set_basename(m_basename);
  msg->set_print(m_print);
}

std::unique_ptr<callback_base> build_dump_model_graph_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackDumpModelGraph&>(
      proto_msg);
  const std::string basename = params.basename();
  const bool print = params.print();
  return std::make_unique<dump_model_graph>(basename.size() == 0 ? "model.dot"
                                                                 : basename,
                                            print);
}

} // namespace callback
} // namespace lbann
