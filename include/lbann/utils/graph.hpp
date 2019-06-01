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
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include "lbann/base.hpp"

namespace lbann {
namespace graph {

/** Print the nodes and edges of a graph to an output stream. */
void print(const std::set<El::Int>& nodes,
           const std::map<El::Int,std::set<El::Int>>& edges,
           std::ostream& os = std::cout);

/** Get nodes adjacent to a given node. */
std::set<El::Int> get_neighbors(El::Int node,
                                const std::map<El::Int,std::set<El::Int>>& edges);

/** @details A closure is a set of nodes with no edges to nodes
 *  outside the set.
 */
bool is_closure(const std::set<El::Int>& nodes,
                const std::map<El::Int,std::set<El::Int>>& edges);

/** Check whether a graph is topologically sorted.
 *
 *  A topologically sorted graph has no edges going from a node to an
 *  earlier node. The graph must be a directed acyclic graph.
 */
bool is_topologically_sorted(const std::set<El::Int>& nodes,
                             const std::map<El::Int,std::set<El::Int>>& edges);

/** Check whether a directed graph is cyclic. */
bool is_cyclic(const std::set<El::Int>& nodes,
               const std::map<El::Int,std::set<El::Int>>& edges);

/** Construct the transpose of a graph.
 *
 *  Reverses the direction of edges in the graph and returns the new
 *  set of edges.
 */
std::map<El::Int,std::set<El::Int>> transpose(const std::set<El::Int>& nodes,
                                              const std::map<El::Int,std::set<El::Int>>& edges);

/** Construct an induced subgraph.
 *
 *  Removes edges to nodes outside the set of nodes and returns the
 *  new set of edges.
 */
std::map<El::Int,std::set<El::Int>> induce_subgraph(const std::set<El::Int>& nodes,
                                                    const std::map<El::Int,std::set<El::Int>>& edges);

/** Perform a breadth-first search starting from a given root node.
 *
 *  The search order is deterministic.
 */
std::vector<El::Int> breadth_first_search(El::Int root,
                                          const std::map<El::Int,std::set<El::Int>>& edges);

/** Perform a depth-first search starting from a given root node.
 *
 *  A depth-first search post-order is returned. The search order is
 *  deterministic.
 */
std::vector<El::Int> depth_first_search(El::Int root,
                                        const std::map<El::Int,std::set<El::Int>>& edges);

/** Topologically sort a graph.
 *
 *  A topologically sorted graph has no edges going from a node to an
 *  earlier node. The sort is deterministic and does not affect graphs
 *  that are already topologically sorted.
 */
std::vector<El::Int> topological_sort(const std::set<El::Int>& nodes,
                                      const std::map<El::Int,std::set<El::Int>>& edges);

/** Construct the condensation of a graph.
 *
 *  The condensation of a graph is constructed by determining the
 *  strongly connected components, i.e. sets of nodes that are
 *  reachable from all nodes in the set, and coalescing them into
 *  single nodes. The condensation is a DAG and will be topologically
 *  sorted.
 */
void condensation(const std::set<El::Int>& nodes,
                  const std::map<El::Int,std::set<El::Int>>& edges,
                  std::map<El::Int,std::set<El::Int>>& components,
                  std::set<El::Int>& condensation_nodes,
                  std::map<El::Int,std::set<El::Int>>& condensation_edges);

}
}
