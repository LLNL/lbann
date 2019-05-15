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

#include "lbann/utils/graph.hpp"
#include "lbann/utils/exception.hpp"
#include <algorithm>
#include <stack>
#include <queue>
#include <unordered_map>

namespace lbann {
namespace graph {

void print(const std::set<El::Int>& nodes,
           const std::map<El::Int,std::set<El::Int>>& edges,
           std::ostream& os) {
  for (const auto& node : nodes) {
    os << "node " << node << " neighbors :";
    for (const auto& neighbor : get_neighbors(node, edges)) {
      os << " " << neighbor;
    }
    os << "\n";
  }
}

std::set<El::Int> get_neighbors(El::Int node,
                                const std::map<El::Int,std::set<El::Int>>& edges) {
  if (edges.count(node) > 0) {
    return edges.at(node);
  } else {
    return {};
  }
}

bool is_closure(const std::set<El::Int>& nodes,
                const std::map<El::Int,std::set<El::Int>>& edges) {
  for (const auto& node : nodes) {
    for (const auto& neighbor : get_neighbors(node, edges)) {
      if (nodes.count(neighbor) == 0) {
        return false;
      }
    }
  }
  return true;
}

bool is_topologically_sorted(const std::set<El::Int>& nodes,
                             const std::map<El::Int,std::set<El::Int>>& edges) {
  if (!is_closure(nodes, edges)) {
    LBANN_ERROR("graph is not a closure");
  }
  for (const auto& node : nodes) {
    const auto& neighbors = get_neighbors(node, edges);
    if (neighbors.size() > 0 && *neighbors.begin() <= node) {
      return false;
    }
  }
  return true;
}

bool is_cyclic(const std::set<El::Int>& nodes,
               const std::map<El::Int,std::set<El::Int>>& edges) {

  // Check that graph is valid
  if (!is_closure(nodes, edges)) {
    LBANN_ERROR("graph is not a closure");
  }

  // Topologically sorted graphs are not cyclic
  if (is_topologically_sorted(nodes, edges)) {
    return false;
  }

  // Perform depth-first searches to detect cycles
  std::unordered_map<El::Int,bool> is_visited, is_sorted;
  std::stack<El::Int> search_stack;
  for (auto&& it = nodes.rbegin(); it != nodes.rend(); ++it) {
    search_stack.push(*it);
  }
  while (!search_stack.empty()) {
    const auto& node = search_stack.top();
    search_stack.pop();
    if (!is_sorted[node]) {
      if (is_visited[node]) {
        is_sorted[node] = true;
      } else {
        is_visited[node] = true;
        search_stack.push(node);
        for (const auto& neighbor : get_neighbors(node, edges)) {
          if (is_visited[neighbor] && !is_sorted[neighbor]) {
            return true;
          }
          search_stack.push(neighbor);
        }
      }
    }
  }
  return false;

}

std::map<El::Int,std::set<El::Int>> transpose(const std::set<El::Int>& nodes,
                                              const std::map<El::Int,std::set<El::Int>>& edges) {
  if (!is_closure(nodes, edges)) {
    LBANN_ERROR("attempted to transpose a graph that is not a closure");
  }
  std::map<El::Int,std::set<El::Int>> transpose_edges;
  for (const auto& node : nodes) {
    for (const auto& neighbor : get_neighbors(node, edges)) {
      transpose_edges[neighbor].insert(node);
    }
  }
  return transpose_edges;
}

std::map<El::Int,std::set<El::Int>> induce_subgraph(const std::set<El::Int>& nodes,
                                                    const std::map<El::Int,std::set<El::Int>>& edges) {
  std::map<El::Int,std::set<El::Int>> induced_edges;
  for (const auto& node : nodes) {
    for (const auto& neighbor : get_neighbors(node, edges)) {
      if (nodes.count(neighbor) > 0) {
        induced_edges[node].insert(neighbor);
      }
    }
  }
  return induced_edges;
}

std::vector<El::Int> breadth_first_search(El::Int root,
                                          const std::map<El::Int,std::set<El::Int>>& edges) {

  // Initialize data structures
  std::unordered_map<El::Int,bool> is_visited;
  std::vector<El::Int> sorted_nodes;
  std::queue<El::Int> search_queue;
  search_queue.push(root);

  // Visit nodes until search queue is exhausted
  while (!search_queue.empty()) {
    const auto& node = search_queue.front();
    search_queue.pop();
    for (const auto& neighbor : get_neighbors(node, edges)) {
      if (!is_visited[neighbor]) {
        is_visited[neighbor] = true;
        sorted_nodes.push_back(neighbor);
        search_queue.push(neighbor);
      }
    }
  }

  // Return list of sorted nodes
  return sorted_nodes;

}

std::vector<El::Int> depth_first_search(El::Int root,
                                        const std::map<El::Int,std::set<El::Int>>& edges) {

  // Initialize data structures
  std::unordered_map<El::Int,bool> is_visited, is_sorted;
  std::vector<El::Int> sorted_nodes;
  std::stack<El::Int> search_stack;
  search_stack.push(root);

  // Visit nodes until search stack is exhausted
  while (!search_stack.empty()) {
    const auto& node = search_stack.top();
    search_stack.pop();
    if (!is_sorted[node]) {
      if (is_visited[node]) {
        // Add node to sorted list if we have already visited
        is_sorted[node] = true;
        sorted_nodes.push_back(node);
      } else {
        // Visit node and add neighbors to search stack
        is_visited[node] = true;
        search_stack.push(node);
        for (const auto& neighbor : get_neighbors(node, edges)) {
          if (!is_visited[neighbor] && !is_sorted[neighbor]) {
            search_stack.push(neighbor);
          }
        }
      }
    }
  }

  // Return list of sorted nodes
  return sorted_nodes;

}


std::vector<El::Int> topological_sort(const std::set<El::Int>& nodes,
                                      const std::map<El::Int,std::set<El::Int>>& edges) {

  // Check that graph is valid
  if (!is_closure(nodes, edges)) {
    LBANN_ERROR("attempted to topologically sort "
                "a graph that is not a closure");
  }
  if (is_cyclic(nodes, edges)) {
    LBANN_ERROR("attempted to topologically sort a cyclic graph");
  }

  // Return original order if already sorted
  if (is_topologically_sorted(nodes, edges)) {
    return std::vector<El::Int>(nodes.begin(), nodes.end());
  }

  // Perform depth-first searches on nodes
  std::stack<El::Int> sorted_stack;
  std::unordered_map<El::Int,bool> is_sorted;
  for (const auto& root : nodes) {
    if (!is_sorted[root]) {
      const auto& dfs = depth_first_search(root, edges);
      for (const auto& node : dfs) {
        if (!is_sorted[node]) {
          is_sorted[node] = true;
          sorted_stack.push(node);
        }
      }
    }
  }

  // Reverse DFS post-order is topologically sorted
  std::vector<El::Int> sorted_nodes;
  while (!sorted_stack.empty()) {
    sorted_nodes.push_back(sorted_stack.top());
    sorted_stack.pop();
  }
  return sorted_nodes;

}

void condensation(const std::set<El::Int>& nodes,
                  const std::map<El::Int,std::set<El::Int>>& edges,
                  std::map<El::Int,std::set<El::Int>>& components,
                  std::set<El::Int>& condensation_nodes,
                  std::map<El::Int,std::set<El::Int>>& condensation_edges) {

  // Initialize data structures for unsorted condensation
  std::unordered_map<El::Int,std::set<El::Int>> unsorted_components;
  std::unordered_map<El::Int,El::Int> unsorted_component_assignments;
  std::set<El::Int> unsorted_condensation_nodes;
  std::map<El::Int,std::set<El::Int>> unsorted_condensation_edges;

  // Find strongly connected components with Kosaraju's algorithm
  // Note: First sort nodes by DFS post-order. Then, pick root nodes
  // in DFS post-order and perform DFS on graph transpose. The first
  // DFS that visits a node determines the strongly connected
  // component it belongs to.
  const auto& transpose_edges = transpose(nodes, edges);
  std::stack<El::Int> dfs_stack;
  std::unordered_map<El::Int,bool> is_sorted, is_condensed;
  for (const auto& root : nodes) {
    if (!is_sorted[root]) {
      for (const auto& node : depth_first_search(root, edges)) {
        if (!is_sorted[node]) {
          is_sorted[node] = true;
          dfs_stack.push(node);
        }
      }
    }
  }
  while (!dfs_stack.empty()) {
    const auto& root = dfs_stack.top();
    dfs_stack.pop();
    if (!is_condensed[root]) {
      const El::Int index = unsorted_condensation_nodes.size();
      unsorted_condensation_nodes.insert(index);
      for (const auto& node : depth_first_search(root, transpose_edges)) {
        if (!is_condensed[node]) {
          is_condensed[node] = true;
          unsorted_component_assignments[node] = index;
          unsorted_components[index].insert(node);
        }
      }
    }
  }

  // Find edges in unsorted condensation
  for (const auto& node : nodes) {
    const auto& unsorted_component = unsorted_component_assignments[node];
    for (const auto& neighbor : get_neighbors(node, edges)) {
      const auto& neighbor_unsorted_component = unsorted_component_assignments[neighbor];
      if (unsorted_component != neighbor_unsorted_component) {
        unsorted_condensation_edges[unsorted_component].insert(neighbor_unsorted_component);
      }
    }
  }

  // Topologically sort condensation
  const auto& sorted_to_unsorted = topological_sort(unsorted_condensation_nodes,
                                                    unsorted_condensation_edges);

  // Record sorted condensation to output
  components.clear();
  condensation_nodes.clear();
  condensation_edges.clear();
  for (size_t i = 0; i < unsorted_condensation_nodes.size(); ++i) {
    condensation_nodes.insert(i);
  }
  std::unordered_map<El::Int,El::Int> unsorted_to_sorted;
  for (const auto& component : condensation_nodes) {
    const auto& unsorted_component = sorted_to_unsorted[component];
    unsorted_to_sorted[unsorted_component] = component;
  }
  for (const auto& unsorted_component : unsorted_condensation_nodes) {
    const auto& component = unsorted_to_sorted[unsorted_component];
    components[component] = unsorted_components[unsorted_component];
    for (const auto& neighbor_unsorted_component : unsorted_condensation_edges[unsorted_component]) {
      condensation_edges[component].insert(unsorted_to_sorted[neighbor_unsorted_component]);
    }
  }

}

}
}
