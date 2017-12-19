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

#include "lbann/utils/graph.hpp"
#include "lbann/utils/exception.hpp"
#include <stack>
#include <algorithm>

namespace lbann {
namespace graph {

bool is_valid(const std::vector<std::set<int>>& graph) {
  const int num_nodes = graph.size();
  for (int node = 0; node < num_nodes; ++node) {
    for (const auto& neighbor : graph[node]) {
      if (0 > neighbor || neighbor >= num_nodes) {
        return false;
      }
    }
  }
  return true;
}

bool is_topologically_sorted(const std::vector<std::set<int>>& graph) {

  // Check that graph is valid
  if (!is_valid(graph)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << "graph is invalid";
    throw lbann_exception(err.str());
  }

  // Visit nodes in order and check for dependency violations
  const int num_nodes = graph.size();
  for (int node = 0; node < num_nodes; ++node) {
    if (graph[node].size() > 0 && *graph[node].begin() <= node) {
      return false;
    }
  }
  return true;

}

bool is_cyclic(const std::vector<std::set<int>>& graph) {

  // Check that graph is valid
  if (!is_valid(graph)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << "graph is invalid";
    throw lbann_exception(err.str());
  }

  // Topologically sorted graphs are not cyclic
  if (is_topologically_sorted(graph)) {
    return false;
  }

  // Perform depth-first searches to detect cycles
  const int num_nodes= graph.size();
  std::vector<bool> is_visited(num_nodes, false), is_sorted(num_nodes, false);
  std::stack<int> search_stack;
  for (int node = graph.size() - 1; node >= 0; --node) {
    search_stack.push(node);
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
        for (const auto& neighbor : graph[node]) {
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

std::vector<std::set<int>> transpose(const std::vector<std::set<int>>& graph) {
  
  // Check that graph is valid
  if (!is_valid(graph)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << "graph is invalid";
    throw lbann_exception(err.str());
  }
  
  // Construct transpose of graph
  const int num_nodes = graph.size();
  std::vector<std::set<int>> graph_transpose(num_nodes);
  for (int node = 0; node < num_nodes; ++node) {
    for (const auto& neighbor : graph[node]) {
      graph_transpose[neighbor].insert(node);
    }
  }
  return graph_transpose;

}

std::vector<int> depth_first_search(const std::vector<std::set<int>>& graph,
                                    int root) {

  // Number of nodes in graph
  const int num_nodes = graph.size();

  // Check that graph is valid
  if (!is_valid(graph)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << "graph is invalid";
    throw lbann_exception(err.str());
  }
  if (0 > root || root >= num_nodes) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "invalid root for depth-first search "
        << "(attempted node " << root << ", "
        << "but there are " << num_nodes << " nodes)";
    throw lbann_exception(err.str());
  }

  // Initialize data structures
  std::vector<bool> is_visited(num_nodes, false), is_sorted(num_nodes, false);
  std::vector<int> sorted_nodes;
  std::stack<int> search_stack;
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
        for (const auto& neighbor : graph[node]) {
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


std::vector<int> topological_sort(const std::vector<std::set<int>>& graph) {

  // Number of nodes in graph
  const int num_nodes = graph.size();

  // Check that graph is valid
  if (!is_valid(graph)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << "graph is invalid";
    throw lbann_exception(err.str());
  }
  if (is_cyclic(graph)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: " << "graph is cyclic";
    throw lbann_exception(err.str());
  }

  // Return original order if already sorted
  if (is_topologically_sorted(graph)) {
    std::vector<int> sequence(num_nodes);
    std::iota(sequence.begin(), sequence.end(), 0);
    return sequence;
  }

  // Perform depth-first searches on nodes
  std::stack<int> sorted_stack;
  std::vector<bool> is_sorted(num_nodes, false);
  for (int root = 0; root < num_nodes; ++root) {
    if (!is_sorted[root]) {
      const auto& dfs = depth_first_search(graph, root);
      for (const auto& node : dfs) {
        if (!is_sorted[node]) {
          is_sorted[node] = true;
          sorted_stack.push(node);
        }
      }
    }
  }

  // Reverse DFS post-order is topologically sorted
  std::vector<int> sorted_nodes;
  while (!sorted_stack.empty()) {
    sorted_nodes.push_back(sorted_stack.top());
    sorted_stack.pop();
  }
  return sorted_nodes;
  
}

void condensation(const std::vector<std::set<int>>& graph,
                  std::vector<std::set<int>>& condensation_nodes,
                  std::vector<std::set<int>>& condensation_edges) {

  // Find strongly connected components with Kosaraju's algorithm
  // Note: First sort nodes by DFS post-order. Then, perform DFS on
  // graph transpose in reverse DFS post-order. Each DFS determines a
  // strongly connected component.
  const int num_nodes = graph.size();
  const auto& graph_transpose = transpose(graph);
  std::stack<int> dfs_stack;
  std::vector<bool> is_sorted(num_nodes, false);
  std::vector<bool> is_condensed(num_nodes, false);
  std::vector<int> component_assignments(num_nodes);
  std::vector<std::set<int>> components;
  for (int root = 0; root < num_nodes; ++root) {
    if (!is_sorted[root]) {
      const auto& dfs = depth_first_search(graph, root);
      for (const auto& node : dfs) {
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
      const auto& dfs = depth_first_search(graph_transpose, root);
      components.emplace_back();
      for (const auto& node : dfs) {
        is_condensed[node] = true;
        component_assignments[node] = components.size() - 1;
        components.back().insert(node);
      }
    }
  }

  // Determine edges in condensation graph
  const int num_components = condensation_nodes.size();
  std::vector<std::set<int>> condensation_graph(num_components);
  for (int node = 0; node < num_nodes; ++node) {
    const auto& component = component_assignments[node];
    for (const auto& neighbor : graph[node]) {
      const auto& neighbor_component = component_assignments[neighbor];
      if (component != neighbor_component) {
        condensation_graph[component].insert(neighbor_component);
      }
    }
  }

  // Topologically sort condensation graph
  const auto& sorted_to_unsorted = topological_sort(condensation_edges);
  std::vector<int> unsorted_to_sorted(num_components);
  for (int sorted_component = 0;
       sorted_component < num_components;
       ++sorted_component) {
    const auto& unsorted_component = sorted_to_unsorted[sorted_component];
    unsorted_to_sorted[unsorted_component] = sorted_component;
  }
  condensation_nodes = std::vector<std::set<int>>(num_components);
  condensation_edges = std::vector<std::set<int>>(num_components);
  for (int sorted_component = 0;
       sorted_component < num_components;
       ++sorted_component) {
    const auto& unsorted_component = sorted_to_unsorted[sorted_component];
    condensation_nodes[sorted_component] = components[unsorted_component];
    for (const auto& neighbor_component : condensation_graph[unsorted_component]) {
      condensation_edges[sorted_component].insert(unsorted_to_sorted[neighbor_component]);
    }
  }

}

}
}
