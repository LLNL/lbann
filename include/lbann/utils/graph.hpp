////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include <vector>
#include <set>

namespace lbann {
namespace graph {

/** Check whether a graph is valid.
 *  The input is interpreted as an adjaceny list for a directed graph.
 */
bool is_valid(const std::vector<std::set<int>>& graph);

/** Check whether a graph is topologically sorted.
 *  The input is interpreted as an adjaceny list for a directed graph.
 */
bool is_topologically_sorted(const std::vector<std::set<int>>& graph);

/** Check whether a graph is cyclic.
 *  The input is interpreted as an adjaceny list for a directed graph.
 */
bool is_cyclic(const std::vector<std::set<int>>& graph);


/** Construct the transpose of a graph.
 *  The input is interpreted as an adjaceny list for a directed graph.
 */
std::vector<std::set<int>> transpose(const std::vector<std::set<int>>& graph);

/** Perform a depth-first search starting from a given root node.
 *  The input is interpreted as an adjaceny list for a directed
 *  graph. The search order is deterministic.
 */
std::vector<int> depth_first_search(const std::vector<std::set<int>>& graph,
                                    int root);

/** Topologically sort a graph.
 *  The input is interpreted as an adjaceny list for a directed
 *  graph. The sort is deterministic.
 */
std::vector<int> topological_sort(const std::vector<std::set<int>>& graph);

/** Construct the condensation of a graph.
 *  In other words, we determine the strongly connected components of
 *  the graph, i.e. sets of nodes that are reachable from each node in
 *  the set. The input is interpreted as an adjaceny list for a
 *  directed graph. The condensation graph is topologically sorted.
 */
void condensation(const std::vector<std::set<int>>& graph,
                  std::vector<std::set<int>>& condensation_nodes,
                  std::vector<std::set<int>>& condensation_edges);

}
}
