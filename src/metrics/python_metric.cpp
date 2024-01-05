////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#include "lbann/metrics/python_metric.hpp"

#include "lbann/io/persist_impl.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"

#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include <Python.h>
#endif

namespace lbann {

template <class Archive>
void python_metric::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("Metric", ::cereal::base_class<metric>(this)),
     CEREAL_NVP(m_name),
     CEREAL_NVP(m_module),
     CEREAL_NVP(m_module_dir),
     CEREAL_NVP(m_function));
}

std::string python_metric::name() const { return m_name; }

std::vector<ViewingLayerPtr> python_metric::get_layer_pointers() const
{
  return std::vector<ViewingLayerPtr>{};
}

void python_metric::set_layer_pointers(std::vector<ViewingLayerPtr> layers)
{
  LBANN_ERROR("Layer pointers should not be set with this metric type");
}

void python_metric::setup(model& m)
{
  metric::setup(m);

#ifndef LBANN_HAS_EMBEDDED_PYTHON
  LBANN_ERROR("Python metrics are not supported. LBANN was not compiled with ",
              "Python.");
#else
  // Make sure Python is running and acquire GIL
  python::global_interpreter_lock gil;

  // Import Python module
  if (!m_module_dir.empty()) {
    auto path = PySys_GetObject("path"); // Borrowed reference
    PyList_Append(path, python::object(m_module_dir));
    python::check_error();
  }
  python::object data_module = PyImport_ImportModule(m_module.c_str());

  // Load function
  m_evaluate_function = PyObject_GetAttrString(data_module, m_function.c_str());

  // Parameter that will be passed to the Python function
  m_model_dir = build_string(get_const_trainer().get_name(), "/", m.get_name());
#endif
}

EvalType python_metric::evaluate(execution_mode mode, int mini_batch_size)
{
  EvalType value = -999.0;
  const auto& start = get_time();
#ifdef LBANN_HAS_EMBEDDED_PYTHON
  {
    python::global_interpreter_lock gil;
    // Call function, convert output reference to EvalType
    python::object result =
      PyObject_CallFunction(m_evaluate_function,
                            "s, i",
                            m_model_dir.c_str(),
                            this->get_comm().get_rank_in_trainer());
    value = static_cast<EvalType>(PyFloat_AsDouble(result));
  }
#endif
  get_evaluate_time() += get_time() - start;
  get_statistics()[mode].add_value(value * mini_batch_size, mini_batch_size);
  return value;
}

bool python_metric::save_to_checkpoint_shared(persist& p)
{
  // write out fields we need to save for model
  if (get_comm().am_trainer_master()) {
    write_cereal_archive<python_metric>(*this,
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

bool python_metric::load_from_checkpoint_shared(persist& p)
{
  load_from_shared_cereal_archive<python_metric>(*this,
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

bool python_metric::save_to_checkpoint_distributed(persist& p)
{
  write_cereal_archive<python_metric>(*this,
                                      p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                      "metrics.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                      "metrics.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

bool python_metric::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive<python_metric>(*this,
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

#define LBANN_CLASS_NAME python_metric
#include <lbann/macros/register_class_with_cereal.hpp>
