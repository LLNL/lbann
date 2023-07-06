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
//
///////////////////////////////////////////////////////////////////////////////
#include "lbann_config.hpp"
#include "lbann/utils/serialize.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <algorithm>
#include <string>
#include <vector>
#include <regex>

#include "lbann/callbacks/profiler_caliper.hpp"

namespace lbann {
namespace callback {


#ifdef LBANN_HAS_CALIPER


using namespace std;

vector<string> split(const string str, const string regex_str)
{
    regex regexz(regex_str);
    vector<string> list(sregex_token_iterator(str.begin(), str.end(), regexz, -1),
                                  sregex_token_iterator());
    return list;
}


bool profiler_caliper::s_autotune=false;
int profiler_caliper::s_tuned_omp_threads=0;

profiler_caliper::profiler_caliper(bool skip_init,bool autotune, int tuned_omp_threads) :
    callback_base(), m_skip_init(skip_init) {
  profiler_caliper::s_autotune = autotune;
  profiler_caliper::s_tuned_omp_threads=tuned_omp_threads;
  struct lbann::adiak_configuration cc;
  adiak::init(NULL);
  adiak::user();
  adiak::launchdate();
  adiak::libraries();
  adiak::cmdline();
  adiak::clustername();
  adiak::jobsize();
  adiak::hostlist();
  adiak::numhosts();
  adiak::walltime();
  adiak::value("lbann_git_version", cc.lbann_git_version);
  adiak::value("cmake_build_type", cc.cmake_build_type);

  adiak::value("compiler_path", cc.compiler);
  cout << "Compiler path: " << cc.compiler << "\n";
  auto tokens = split(cc.compiler, "/");
  string compiler_exec = tokens.back();
  adiak::value("compiler_version", cc.compiler_version);
  string compiler = compiler_exec + "-" + cc.compiler_version;
  cout << "Compiler: " << compiler << "\n";
  adiak::value("compiler", compiler.c_str());
  auto tsize = tokens.size();
  if (tsize >= 4) {
    // pickup path version <compiler-version-hash|date>/mpispec/bin/exec
    string path_version = tokens[tsize-4];
    cout << "Compiler path version: " << path_version << "\n";
    auto s = split(path_version,"-");
    if (s.size() >= 2) {
      string path_version_short = s[0] + "-" + s[1];
      cout << "Compiler path version short: " << path_version_short << "\n";
      adiak::value("Compiler_path_version",path_version_short.c_str());
    }
  }

  adiak::value("compiler_flags", cc.compiler_flags);
  if (!strcmp(cc.cmake_build_type, "Release"))
    adiak::value("compiler_flags_release", cc.compiler_flags_release);
  else if (!strcmp(cc.cmake_build_type, "RelWithDebInfo"))
    adiak::value("compiler_flags_relwithdebinfo", cc.compiler_flags_relwithdebinfo);
  else if (!strcmp(cc.cmake_build_type, "Debug"))
    adiak::value("compiler_flags_debug", cc.compiler_flags_debug);

  if (strlen(cc.cuda_compiler_version) > 0) {
    adiak::value("cuda_compiler_version", cc.cuda_compiler_version);
    adiak::value("cuda_flags", cc.cuda_flags);
    adiak::value("cuda_flags_release", cc.cuda_flags_release);
  }

  // Openmp section
  // todo get lib : e.g libomp,libiomp5,libgomp etc  : parse adiak::libraries via tool callback
  // note version map only goes to 5.1; revise as needed
#if defined(_OPENMP)
  std::unordered_map<unsigned,std::string> map{
    {200505,"2.5"},{200805,"3.0"},{201107,"3.1"},{201307,"4.0"},{201511,"4.5"},{201811,"5.0"},{202011,"5.1"}};
  string strval = map.at(_OPENMP);
  adiak::value("omp_version",strval.c_str());
  strval = std::to_string(omp_get_max_threads());
  adiak::value("omp_max_threads",strval.c_str());
#endif

  if (!m_skip_init) {
    start();
  }
}

profiler_caliper::~profiler_caliper() {}

template <class Archive>
void profiler_caliper::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_skip_init));
}

void profiler_caliper::on_epoch_begin(model *m) {
  const auto& c = static_cast<SGDExecutionContext&>(m->get_execution_context());
  static int epochs = 0;
  epochs++;

  // Skip the first epoch
  if (m_skip_init && c.get_epoch() == 1) {
    epochs--;
    start();
  }
  string strval = std::to_string(epochs);
  adiak::value("epochs_timed",strval.c_str()); // this will get updated every epoch until terminate
  CALI_MARK_BEGIN("epoch");
}

void profiler_caliper::on_epoch_end(model *m) {
  CALI_MARK_END("epoch");
}

void profiler_caliper::on_validation_begin(model *m) {
  CALI_MARK_BEGIN("validation");
}

void profiler_caliper::on_validation_end(model *m) {
  CALI_MARK_END("validation");
}

void profiler_caliper::on_test_begin(model *m) {
  CALI_MARK_BEGIN("test");
}

void profiler_caliper::on_test_end(model *m) {
  CALI_MARK_END("test");
}

void profiler_caliper::on_batch_begin(model *m) {
  CALI_MARK_BEGIN("batch");
}

void profiler_caliper::on_batch_end(model *m) {
  CALI_MARK_END("batch");
}

void profiler_caliper::on_batch_evaluate_begin(model *m) {
  CALI_MARK_BEGIN("batch_evaluate");
}

void profiler_caliper::on_batch_evaluate_end(model *m) {
  CALI_MARK_END("batch_evaluate");
}

void profiler_caliper::on_forward_prop_begin(model *m) {
  CALI_MARK_BEGIN("forward_prop");
}

void profiler_caliper::on_forward_prop_end(model *m) {
  CALI_MARK_END("forward_prop");
}

void profiler_caliper::on_evaluate_forward_prop_begin(model *m) {
  CALI_MARK_BEGIN("evaluate_forward_prop");
}

void profiler_caliper::on_evaluate_forward_prop_end(model *m) {
  CALI_MARK_END("evaluate_forward_prop");
}

void profiler_caliper::on_backward_prop_begin(model *m) {
  CALI_MARK_BEGIN("backward_prop");
}

void profiler_caliper::on_backward_prop_end(model *m) {
  CALI_MARK_END("backward_prop");
}

void profiler_caliper::on_optimize_begin(model *m) {
  CALI_MARK_BEGIN("optimize");
}

void profiler_caliper::on_optimize_end(model *m) {
  CALI_MARK_END("optimize");
}

void profiler_caliper::on_forward_prop_begin(model *m, Layer *l) {
  std::string mark = "fw::" + l->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_forward_prop_end(model *m, Layer *l) {
  std::string mark = "fw::" + l->get_name();
  CALI_MARK_END(mark.c_str());
}

void profiler_caliper::on_evaluate_forward_prop_begin(model *m, Layer *l) {
  std::string mark = "eval_fw::" + l->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_evaluate_forward_prop_end(model *m, Layer *l) {
  std::string mark = "eval_fw::" + l->get_name();
  CALI_MARK_END(mark.c_str());
}

void profiler_caliper::on_backward_prop_begin(model *m, Layer *l) {
  std::string mark = "bw::" + l->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_backward_prop_end(model *m, Layer *l) {
  std::string mark = "bw::" + l->get_name();
  CALI_MARK_END(mark.c_str());
}

void profiler_caliper::on_optimize_begin(model *m, weights *w) {
  std::string mark = "opt::" + w->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_optimize_end(model *m, weights *w) {
  std::string mark = "opt::" + w->get_name();
  CALI_MARK_END(mark.c_str());
}

#endif // LBANN_HAS_CALIPER


std::unique_ptr<callback_base>
build_profiler_caliper_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
#ifdef LBANN_HAS_CALIPER
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackProfilerCaliper&>(proto_msg);
  return make_unique<profiler_caliper>(params.skip_init(),params.autotune(),params.tuned_omp_threads());
#else
  LBANN_ERROR("CallbackProfileCaliper is not available; Caliper support not detected.");
#endif
}


} // namespace callback
} // namespace lbann

#ifdef LBANN_HAS_CALIPER
#define LBANN_CLASS_NAME callback::profiler_caliper
#define LBANN_CLASS_LIBNAME callback_profiler_caliper
#include <lbann/macros/register_class_with_cereal.hpp>
#endif
