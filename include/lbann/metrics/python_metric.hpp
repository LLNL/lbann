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

#ifndef LBANN_METRIC_PYTHON_METRIC_HPP
#define LBANN_METRIC_PYTHON_METRIC_HPP

#include "lbann/metrics/metric.hpp"

#ifdef LBANN_HAS_EMBEDDED_PYTHON
#include "lbann/utils/python.hpp"
#endif

namespace lbann {

/** @brief A metric that receives its value from the return value of a Python
 *  function.
 *
 *  Similarly to the Python data reader, this metric will use the built-in
 *  Python interpreter to import a module and call a function within it.
 *  The module (given as ``module``) should be accessible by trainer ranks and
 *  located in the ``module_dir`` path. For example, if the Python file is in
 *  ``/path/to/myfile.py``, set ``module_dir`` to ``/path/to`` and ``module``
 *  to ``myfile``.
 *
 *  Within the imported Python module, the function ``function`` will be called.
 *  The function is expected to accept one string argument, representing
 *  the experiment path (trainer and model names), and one int argument
 *  representing the rank. It returns one float value (cast to ``EvalType`` in
 *  LBANN). For example, for a function called "evaluate", the expected
 *  prototype would be:
 *  ``def evaluate(experiment_path: str, rank: int) -> float: ...``
 *
 *  Note that this metric is only available if LBANN was compiled with Python
 *  support.
 */
class python_metric : public metric
{

public:
  python_metric(lbann_comm* comm = nullptr,
                std::string name = "",
                std::string module = "",
                std::string module_dir = "",
                std::string function = "")
    : metric(comm),
      m_name(name),
      m_module(module),
      m_module_dir(module_dir),
      m_function(function)
  {}
  python_metric(const python_metric& other) = default;
  python_metric& operator=(const python_metric& other) = default;
  virtual ~python_metric() = default;
  python_metric* copy() const override { return new python_metric(*this); }

  /** Return a string name for this metric. */
  std::string name() const override;

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** Get list of pointers to layers. */
  std::vector<ViewingLayerPtr> get_layer_pointers() const override;
  /** Set list of pointers to layers. */
  void set_layer_pointers(std::vector<ViewingLayerPtr> layers) override;

  /** Save metric state to checkpoint. */
  bool save_to_checkpoint_shared(persist& p) override;
  /** Load metric state from checkpoint. */
  bool load_from_checkpoint_shared(persist& p) override;

  bool save_to_checkpoint_distributed(persist& p) override;
  bool load_from_checkpoint_distributed(persist& p) override;

protected:
  void setup(model& m) override;
  EvalType evaluate(execution_mode mode, int mini_batch_size) override;

private:
  /** Descriptive name for metric. */
  std::string m_name;

  /** Python module name to load. */
  std::string m_module;

  /** Directory in which the Python module resides. */
  std::string m_module_dir;

  /** Python function to call in module. */
  std::string m_function;

#ifdef LBANN_HAS_EMBEDDED_PYTHON
  python::object m_evaluate_function;
  std::string m_model_dir;
#endif
};

} // namespace lbann

#endif // LBANN_METRIC_PYTHON_METRIC_HPP
