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
#include <map>

namespace lbann {
namespace graph {

/** Get nodes adjacent to a given node. */
std::set<int> get_neighbors(int node,
                            const std::map<int,std::set<int>>& edges);

/** Check whether a graph is a closure.
 *  A closure is a set of nodes with no edges to nodes outside the
 *  set.
 */
bool is_closure(const std::set<int>& nodes,
                const std::map<int,std::set<int>>& edges);

/** Check whether a graph is topologically sorted.
 *  A topologically sorted graph has no edges going from a node to an
 *  earlier node. The graph must be a directed acyclic graph.
 */
bool is_topologically_sorted(const std::set<int>& nodes,
                             const std::map<int,std::set<int>>& edges);

/** Check whether a directed graph is cyclic. */
bool is_cyclic(const std::set<int>& nodes,
               const std::map<int,std::set<int>>& edges);

/** Construct the transpose of a graph.
 *  Reverses the direction of edges in the graph and returns the new
 *  set of edges.
 */
std::map<int,std::set<int>> transpose(const std::set<int>& nodes,
                                      const std::map<int,std::set<int>>& edges);

/** Construct an induced subgraph.
 *  Removes edges to nodes outside the set of nodes and returns the
 *  new set of edges.
 */
std::map<int,std::set<int>> induce_subgraph(const std::set<int>& nodes,
                                            const std::map<int,std::set<int>>& edges);

/** Perform a depth-first search starting from a given root node.
 *  The search order is deterministic.
 */
std::vector<int> depth_first_search(const std::map<int,std::set<int>>& edges,
                                    int root);

/** Topologically sort a graph.
 *  A topologically sorted graph has no edges going from a node to an
 *  earlier node. The sort is deterministic and does not affect graphs
 *  that are already topologically sorted.
 */
std::vector<int> topological_sort(const std::set<int>& nodes,
                                  const std::map<int,std::set<int>>& edges);

/** Construct the condensation of a graph.
 *  The condensation of a graph is constructed by determining the
 *  strongly connected components, i.e. sets of nodes that are
 *  reachable from all nodes in the set, and coalescing them into
 *  single nodes. The condensation is a DAG and will be topologically
 *  sorted.
 */
void condensation(const std::set<int>& nodes,
                  const std::map<int,std::set<int>>& edges,
                  std::map<int,std::set<int>>& components,
                  std::set<int>& condensation_nodes,
                  std::vector<std::set<int>>& condensation_edges);

}
}
