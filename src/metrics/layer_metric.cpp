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

#include "lbann/metrics/layer_metric.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

layer_metric::layer_metric(lbann_comm* comm,
                           std::string name_,
                           std::string unit)
  : metric(comm), m_name(name_), m_unit(unit)
{}

template <class Archive>
void layer_metric::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("Metric", ::cereal::base_class<metric>(this)),
     CEREAL_NVP(m_name),
     CEREAL_NVP(m_unit),
     CEREAL_NVP(m_layer));
}

std::string layer_metric::name() const
{
  if (!m_name.empty()) {
    return m_name;
  }
  else if (m_layer.expired()) {
    return "uninitialized layer metric";
  }
  else {
    return m_layer.lock()->get_name();
  }
}

void layer_metric::set_layer(ViewingLayerPtr l) { m_layer = std::move(l); }
Layer& layer_metric::get_layer()
{
  // Idiom from Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers.
  return *(
    const_cast<Layer*>(&static_cast<const layer_metric&>(*this).get_layer()));
}
const Layer& layer_metric::get_layer() const
{
  if (m_layer.expired()) {
    std::stringstream err;
    err << "attempted to get the layer corresponding to "
        << "layer metric \"" << name() << "\", "
        << "but no such layer has been set";
    LBANN_ERROR(err.str());
  }
  return *m_layer.lock();
}

std::vector<ViewingLayerPtr> layer_metric::get_layer_pointers() const
{
  auto layer_pointers = metric::get_layer_pointers();
  layer_pointers.push_back(m_layer);
  return layer_pointers;
}

void layer_metric::set_layer_pointers(std::vector<ViewingLayerPtr> layers)
{
  metric::set_layer_pointers(
    std::vector<ViewingLayerPtr>(layers.begin(), layers.end() - 1));
  m_layer = layers.back();
}

void layer_metric::setup(model& m)
{
  metric::setup(m);
  get_evaluation_layer();
}

EvalType layer_metric::evaluate(execution_mode mode, int mini_batch_size)
{
  const auto& start = get_time();
  auto value =
    dynamic_cast<abstract_evaluation_layer<DataType>&>(get_evaluation_layer())
      .get_value(false);
  get_evaluate_time() += get_time() - start;
  if (m_unit == "%") {
    value *= 100;
  }
  get_statistics()[mode].add_value(value * mini_batch_size, mini_batch_size);
  return value;
}

/*abstract_evaluation_*/ Layer& layer_metric::get_evaluation_layer()
{
  auto& l = get_layer();
  auto* eval = dynamic_cast<abstract_evaluation_layer<DataType>*>(&l);
  if (eval == nullptr) {
    std::stringstream err;
    err << "attempted to get the evaluation layer corresponding to "
        << "layer metric \"" << name() << "\", "
        << "but it currently corresponds to " << l.get_type() << " layer \""
        << l.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return *eval;
}

bool layer_metric::save_to_checkpoint_shared(persist& p)
{
  // write out fields we need to save for model
  if (get_comm().am_trainer_master()) {
    write_cereal_archive<layer_metric>(*this,
                                       p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                       "metrics.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                       "metrics.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  return true;
}

bool layer_metric::load_from_checkpoint_shared(persist& p)
{
  load_from_shared_cereal_archive<layer_metric>(*this,
                                                p,
                                                get_comm(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                                "metrics.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                                "metrics.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

bool layer_metric::save_to_checkpoint_distributed(persist& p)
{
  write_cereal_archive<layer_metric>(*this,
                                     p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                     "metrics.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                     "metrics.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

bool layer_metric::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive<layer_metric>(*this,
                                    p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                    "metrics.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                    "metrics.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

} // namespace lbann

#define LBANN_CLASS_NAME layer_metric
#include <lbann/macros/register_class_with_cereal.hpp>
