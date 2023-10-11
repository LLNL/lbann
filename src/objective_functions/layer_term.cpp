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

#include "lbann/objective_functions/layer_term.hpp"
#include "lbann/proto/objective_functions.pb.h"
#include "lbann/utils/serialize.hpp"

namespace lbann {

layer_term::layer_term(EvalType scale_factor)
  : objective_function_term(scale_factor)
{}

template <typename ArchiveT>
void layer_term::serialize(ArchiveT& ar)
{
  ar(::cereal::make_nvp("ObjectiveFunctionTerm",
                        ::cereal::base_class<objective_function_term>(this)));
}

void layer_term::set_layer(ViewingLayerPtr l)
{
  std::vector<ViewingLayerPtr> ptrs;
  ptrs.emplace_back(std::move(l));
  set_layer_pointers(ptrs);
}

Layer& layer_term::get_layer()
{
  // Idiom from Item 3, p. 23 in "Effective C++", 3rd ed., by Scott Meyers.
  return *(
    const_cast<Layer*>(&static_cast<const layer_term&>(*this).get_layer()));
}
const Layer& layer_term::get_layer() const
{
  const auto layer_pointers = get_layer_pointers();
  if (layer_pointers.empty() || layer_pointers.front().expired()) {
    LBANN_ERROR("attempted to get the layer corresponding to "
                "an objective function layer term, "
                "but no such layer has been set");
  }
  return *layer_pointers.front().lock();
}

/*abstract_evaluation_*/ Layer& layer_term::get_evaluation_layer()
{
  auto& l = get_layer();
  auto* eval = dynamic_cast<abstract_evaluation_layer<DataType>*>(&l);
  if (eval == nullptr) {
    std::stringstream err;
    err << "attempted to get the evaluation layer corresponding to "
        << "an objective function layer term, "
        << "but the layer term currently corresponds to " << l.get_type()
        << " layer \"" << l.get_name() << "\"";
    LBANN_ERROR(err.str());
  }
  return *eval;
}

void layer_term::setup(model& m)
{
  objective_function_term::setup(m);
  auto& eval =
    dynamic_cast<abstract_evaluation_layer<DataType>&>(get_evaluation_layer());
  eval.set_scale(m_scale_factor);
  eval.set_amp_scale(m_amp_scale_factor);
  // get_evaluation_layer().set_scale(m_scale_factor);
}

void layer_term::start_evaluation() {}

EvalType layer_term::finish_evaluation()
{
  if (m_scale_factor == EvalType(0)) {
    return EvalType(0);
  }
  auto& eval =
    dynamic_cast<abstract_evaluation_layer<DataType>&>(get_evaluation_layer());
  eval.set_scale(m_scale_factor);
  eval.set_amp_scale(m_amp_scale_factor);
  return eval.get_value();
}

void layer_term::differentiate()
{
  auto& eval =
    dynamic_cast<abstract_evaluation_layer<DataType>&>(get_evaluation_layer());
  eval.set_scale(m_scale_factor);
  eval.set_amp_scale(m_amp_scale_factor);
  // get_evaluation_layer().set_scale(m_scale_factor);
}

void layer_term::write_specific_proto(
  lbann_data::ObjectiveFunction& proto) const
{
  auto* term_msg = proto.add_layer_term();
  term_msg->set_scale_factor(this->m_scale_factor);
  term_msg->set_layer(this->get_layer().get_name());
}

} // namespace lbann

#define LBANN_CLASS_NAME layer_term
#include <lbann/macros/register_class_with_cereal.hpp>
