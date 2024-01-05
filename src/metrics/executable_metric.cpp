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

#include <cstdio>  // For popen, pclose, fread
#include <cstdlib> // For strtod

#include "lbann/io/persist_impl.hpp"
#include "lbann/metrics/executable_metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

template <class Archive>
void executable_metric::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("Metric", ::cereal::base_class<metric>(this)),
     CEREAL_NVP(m_name),
     CEREAL_NVP(m_filename),
     CEREAL_NVP(m_other_args));
}

std::string executable_metric::name() const { return m_name; }

std::vector<ViewingLayerPtr> executable_metric::get_layer_pointers() const
{
  return std::vector<ViewingLayerPtr>{};
}

void executable_metric::set_layer_pointers(std::vector<ViewingLayerPtr> layers)
{
  LBANN_ERROR("Layer pointers should not be set with this metric type");
}

void executable_metric::setup(model& m)
{
  metric::setup(m);
  m_cmd = build_string(m_filename,
                       " ",
                       m_other_args,
                       " ",
                       get_const_trainer().get_name(),
                       "/",
                       m.get_name());
}

static inline EvalType spawn_process_and_read_output(const char* cmd)
{
  // More than enough to read one EvalType value (I hope)
  char buffer[2048];

  FILE* fp = popen(cmd, "r");
  size_t read = fread(buffer, sizeof(char), 2048, fp);
  pclose(fp);

  if (read == 2048) { // Buffer is potentially too long
    std::string as_str;
    as_str.assign(buffer, 2048);
    LBANN_ERROR("Process output of \"",
                cmd,
                "\" is too long. Contents of ",
                "output start with:\n",
                as_str);
  }
  buffer[read] = '\0'; // NULL terminator

  char* endptr;
  double result = strtod(buffer, &endptr);
  if (endptr == buffer) { // Invalid number
    LBANN_ERROR("Process output of \"",
                cmd,
                "\" is not a valid number. ",
                "Output:\n",
                std::string(buffer));
  }

  return static_cast<EvalType>(result);
}

EvalType executable_metric::evaluate(execution_mode mode, int mini_batch_size)
{
  const auto& start = get_time();
  EvalType value = spawn_process_and_read_output(m_cmd.c_str());
  get_evaluate_time() += get_time() - start;
  get_statistics()[mode].add_value(value * mini_batch_size, mini_batch_size);
  return value;
}

bool executable_metric::save_to_checkpoint_shared(persist& p)
{
  // write out fields we need to save for model
  if (get_comm().am_trainer_master()) {
    write_cereal_archive<executable_metric>(*this,
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

bool executable_metric::load_from_checkpoint_shared(persist& p)
{
  load_from_shared_cereal_archive<executable_metric>(*this,
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

bool executable_metric::save_to_checkpoint_distributed(persist& p)
{
  write_cereal_archive<executable_metric>(*this,
                                          p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                          "metrics.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                          "metrics.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return true;
}

bool executable_metric::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive<executable_metric>(*this,
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

#define LBANN_CLASS_NAME executable_metric
#include <lbann/macros/register_class_with_cereal.hpp>
