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
//
// profiling .hpp .cpp - Various routines for interfacing with profilers
///////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/profiling.hpp"
#include "lbann/base.hpp"
#include "lbann/utils/exception.hpp"
// For get_current_comm, which is here for some reason.
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/options.hpp"

#if defined(LBANN_SCOREP)
#include <scorep/SCOREP_User.h>
#elif defined(LBANN_NVPROF)
#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "lbann/utils/gpu/helpers.hpp"
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#endif

#if defined(LBANN_HAS_ROCTRACER)
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctx.h>
#endif

#ifdef LBANN_HAS_CALIPER
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

#include "adiak_config.hpp"
#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif // defined(_OPENMP)
#endif // LBANN_HAS_CALIPER

namespace {
bool profiling_started = false;
}

namespace lbann {

// If Caliper is available, it is used unilaterally. If it were
// composed with other annotation APIs (nvtx, roctx, etc), one could
// end up with double marked regions. Instead, ensure that Caliper has
// been configured with these tools and access them via Caliper.
#if defined(LBANN_HAS_CALIPER)
namespace {
std::vector<std::string> split(std::string const str,
                               std::string const regex_str)
{
  std::regex regexz(regex_str);
  std::vector<std::string> list(
    std::sregex_token_iterator(str.begin(), str.end(), regexz, -1),
    std::sregex_token_iterator());
  return list;
}

std::string as_lowercase(std::string str)
{
  std::transform(cbegin(str),
                 cend(str),
                 begin(str),
                 [](unsigned char c) { return ::tolower(c); });
  return str;
}

// FIXME (trb): There's potentially a good amount of other metadata
// that might be outside the scope of this "profiling" file. This
// might be better factored out of here and managed independently of
// caliper-based profiling. E.g., much of this information would be
// useful if it were managed by the LBANN executable and dropped in
// the working directory as a general artifact of the run.
void do_adiak_init()
{
  struct adiak_configuration const cc;
  adiak::init(utils::get_current_comm().get_world_comm().GetMPIComm());
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

  // Compiler information:
  auto const tokens = split(cc.compiler, "/");
  auto const tsize = tokens.size();
  std::string const compiler_exec = tokens.back();
  std::string const compiler = compiler_exec + "-" + cc.compiler_version;
  adiak::value("compiler", compiler);

  adiak::value("compiler_path", cc.compiler);
  adiak::value("compiler_version", cc.compiler_version);
  // FIXME: How robust is this?? Seems like "not very" to me.
  if (tsize >= 4) {
    // pickup path version <compiler-version-hash|date>/mpispec/bin/exec
    std::string const path_version = tokens[tsize - 4];
    std::cout << "Compiler path version: " << path_version << "\n";
    auto const s = split(path_version, "-");
    if (s.size() >= 2) {
      std::string const path_version_short = s[0] + "-" + s[1];
      adiak::value("compiler_path_version", path_version_short);
    }
  }

  // Flag information
  auto const build_type = as_lowercase(cc.cmake_build_type);
  adiak::value("compiler_flags", cc.compiler_flags);
  if (build_type == "release")
    adiak::value("compiler_flags_release", cc.compiler_flags_release);
  else if (build_type == "relwithdebinfo")
    adiak::value("compiler_flags_relwithdebinfo",
                 cc.compiler_flags_relwithdebinfo);
  else if (build_type == "debug")
    adiak::value("compiler_flags_debug", cc.compiler_flags_debug);

#ifdef LBANN_HAS_CUDA
  if (strlen(cc.cuda_compiler) > 0) {
    adiak::value("cuda_compiler", cc.cuda_compiler);
    adiak::value("cuda_compiler_version", cc.cuda_compiler_version);
    adiak::value("cuda_flags", cc.cuda_flags);
    if (build_type == "release")
      adiak::value("cuda_flags_release", cc.cuda_flags_release);
    else if (build_type == "relwithdebinfo")
      adiak::value("cuda_flags_relwithdebinfo", cc.cuda_flags_relwithdebinfo);
    else if (build_type == "debug")
      adiak::value("cuda_flags_debug", cc.cuda_flags_debug);
  }
#endif // LBANN_HAS_CUDA
#ifdef LBANN_HAS_ROCM
  if (strlen(cc.hip_compiler) > 0) {
    adiak::value("hip_compiler", cc.hip_compiler);
    adiak::value("hip_compiler_version", cc.hip_compiler_version);
    adiak::value("hip_flags", cc.hip_flags);
    if (build_type == "release")
      adiak::value("hip_flags_release", cc.hip_flags_release);
    else if (build_type == "relwithdebinfo")
      adiak::value("hip_flags_relwithdebinfo", cc.hip_flags_relwithdebinfo);
    else if (build_type == "debug")
      adiak::value("hip_flags_debug", cc.hip_flags_debug);
  }
#endif // LBANN_HAS_ROCM

  // Openmp section
  // todo get lib : e.g libomp,libiomp5,libgomp etc  : parse adiak::libraries
  // via tool callback note version map only goes to 5.1; revise as needed
#if defined(_OPENMP)
  std::unordered_map<unsigned, std::string> map{{200505, "2.5"},
                                                {200805, "3.0"},
                                                {201107, "3.1"},
                                                {201307, "4.0"},
                                                {201511, "4.5"},
                                                {201811, "5.0"},
                                                {202011, "5.1"},
                                                {202111, "5.2"}};
  adiak::value("omp_version", map.at(_OPENMP));
  adiak::value("omp_max_threads", omp_get_max_threads());
#endif
}
void do_adiak_finalize() {
  adiak::fini();
}

cali::ConfigManager cali_mgr;
bool caliper_initialized = false;
}// namespace
void prof_start()
{
  if (caliper_initialized) {
    LBANN_ERROR("Cannot reinitialize Caliper");
  }
  do_adiak_init();

  auto& arg_parser = global_argument_parser();
  cali_mgr.add(
    arg_parser.get<std::string>(LBANN_OPTION_CALIPER_CONFIG).c_str());
  cali_mgr.start();

  profiling_started = true;
  caliper_initialized = true;
}
void prof_stop()
{
  cali_mgr.stop();
  cali_mgr.flush();
  do_adiak_finalize();
  profiling_started = false;
}
void prof_region_begin(const char* s, int, bool)
{
  if (!profiling_started) return;
  cali_begin_region(s);
}
void prof_region_end(const char* s, bool)
{
  if (!profiling_started) return;
  cali_end_region(s);
}
#elif defined(LBANN_SCOREP)
void prof_start()
{
  profiling_started = true;
  return;
}
void prof_stop() { return; }
void prof_region_begin(const char* s, int, bool)
{
  SCOREP_USER_REGION_BY_NAME_BEGIN(s, SCOREP_USER_REGION_TYPE_COMMON);
  return;
}
void prof_region_end(const char* s, bool)
{
  SCOREP_USER_REGION_BY_NAME_END(s);
  return;
}
#elif defined(LBANN_NVPROF)
void prof_start()
{
  CHECK_CUDA(cudaProfilerStart());
  profiling_started = true;
}
void prof_stop()
{
  CHECK_CUDA(cudaProfilerStop());
  profiling_started = false;
}
void prof_region_begin(const char* s, int c, bool sync)
{
  if (!profiling_started)
    return;
  if (sync) {
    hydrogen::gpu::SynchronizeDevice();
  }
  // Doesn't work with gcc 4.9
  // nvtxEventAttributes_t ev = {0};
  nvtxEventAttributes_t ev;
  memset(&ev, 0, sizeof(nvtxEventAttributes_t));
  ev.version = NVTX_VERSION;
  ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  ev.colorType = NVTX_COLOR_ARGB;
  ev.color = c;
  ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ev.message.ascii = s;
  nvtxRangePushEx(&ev);
}
void prof_region_end(const char*, bool sync)
{
  if (!profiling_started)
    return;
  if (sync) {
    hydrogen::gpu::SynchronizeDevice();
  }
  nvtxRangePop();
}
#elif defined(LBANN_HAS_ROCTRACER)
void prof_start()
{
  roctracer_start();
  profiling_started = true;
}
void prof_stop()
{
  roctracer_stop();
  profiling_started = false;
}
void prof_region_begin(const char* s, int, bool sync)
{
  if (!profiling_started)
    return;
  if (sync) {
    hydrogen::gpu::SynchronizeDevice();
  }
  LBANN_ASSERT(0 <= roctxRangePush(s));
}
void prof_region_end(const char*, bool sync)
{
  if (!profiling_started)
    return;
  if (sync) {
    hydrogen::gpu::SynchronizeDevice();
  }
  LBANN_ASSERT(0 <= roctxRangePop());
}
#else
void prof_start()
{
  profiling_started = true;
  return;
}
void prof_stop()
{
  profiling_started = false;
  return;
}
void prof_region_begin(const char*, int, bool) { return; }
void prof_region_end(const char*, bool) { return; }
#endif

static int next_color() noexcept
{
  static int idx = -1;
  idx = (idx + 1) % num_prof_colors;
  return prof_colors[idx];
}

ProfRegion::ProfRegion(char const* name, bool sync)
  : ProfRegion{name, next_color(), sync}
{}

ProfRegion::ProfRegion(char const* name, int color, bool sync)
  : m_name{name}, m_sync{sync}
{
  prof_region_begin(m_name, color, m_sync);
}

ProfRegion::~ProfRegion() { prof_region_end(m_name, m_sync); }

} // namespace lbann
