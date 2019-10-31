/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_TREE_VISUALIZATION_HPP
#define ARBORX_DETAILS_TREE_VISUALIZATION_HPP

#include <ArborX_DetailsTreeTraversal.hpp>

#include <Kokkos_View.hpp>

namespace ArborX
{

template <typename DeviceType>
class BoundingVolumeHierarchy;

namespace Details
{
std::ostream &operator<<(std::ostream &os, Point const &p)
{
  os << "(" << p[0] << "," << p[1] << ")";
  return os;
}

template <typename DeviceType>
struct TreeVisualization
{
#if defined(KOKKOS_ENABLE_SERIAL)
  static_assert(
      std::is_same<typename DeviceType::execution_space, Kokkos::Serial>::value,
      "tree visualization only available in serial");
#else
  static_assert(
      false,
      "tree visualization not available when Kokkos serial is not enabled");
#endif
  struct TreeAccess
  {
    KOKKOS_INLINE_FUNCTION
    static Node const *getLeaf(BoundingVolumeHierarchy<DeviceType> const &bvh,
                               size_t index)
    {
      auto leaf_nodes = bvh.getLeafNodes();
      Node const *first = leaf_nodes.data();
      Node const *last = first + static_cast<ptrdiff_t>(leaf_nodes.size());
      for (; first != last; ++first)
        if (index == first->getLeafPermutationIndex())
          return first;
      return nullptr;
    }

    KOKKOS_INLINE_FUNCTION
    static int getIndex(Node const *node,
                        BoundingVolumeHierarchy<DeviceType> const &bvh)
    {
      return node->isLeaf() ? node->getLeafPermutationIndex()
                            : node - bvh.getRoot();
    }

    KOKKOS_INLINE_FUNCTION
    static Node const *getRoot(BoundingVolumeHierarchy<DeviceType> const &bvh)
    {
      return bvh.getRoot();
    }

    template <typename Tree>
    KOKKOS_INLINE_FUNCTION static typename Tree::bounding_volume_type
    getBoundingVolume(Node const *node, Tree const &tree)
    {
      return tree.getBoundingVolume(node);
    }
  };

  template <typename Tree>
  static std::string getNodeLabel(Node const *node, Tree const &tree)
  {
    auto const node_is_leaf = node->isLeaf();
    auto const node_index = TreeAccess::getIndex(node, tree);
    std::string label = node_is_leaf ? "l" : "i";
    label.append(std::to_string(node_index));
    return label;
  }

  template <typename Tree>
  static std::string getNodeAttributes(Node const *node, Tree const &)
  {
    return node->isLeaf() ? "[leaf]" : "[internal]";
  }

  template <typename Tree>
  static std::string getEdgeAttributes(Node const * /*parent*/,
                                       Node const *child, Tree const &)
  {
    return child->isLeaf() ? "[pendant]" : "[edge]";
  }

  // Produces node and edges statements to be listed for a graph in DOT
  // format:
  // ```
  // digraph g {
  //   root = i0;
  //   <paste node and edges statements here>
  // }
  // ```
  struct GraphvizVisitor
  {
    std::ostream &_os;

    template <typename Tree>
    void visit(Node const *node, Tree const &tree) const
    {
      visitNode(node, tree);
      visitEdgesStartingFromNode(node, tree);
    }

    template <typename Tree>
    void visitNode(Node const *node, Tree const &tree) const
    {
      auto const node_label = getNodeLabel(node, tree);
      auto const node_attributes = getNodeAttributes(node, tree);

      _os << "    " << node_label << " " << node_attributes << ";\n";
    }

    template <typename Tree>
    void visitEdgesStartingFromNode(Node const *node, Tree const &tree) const
    {
      auto const node_label = getNodeLabel(node, tree);
      auto const node_is_internal = !node->isLeaf();

      if (node_is_internal)
        for (Node const *child : {node->children.first, node->children.second})
        {
          auto const child_label = getNodeLabel(child, tree);
          auto const edge_attributes = getEdgeAttributes(node, child, tree);

          _os << "    " << node_label << " -> " << child_label << " "
              << edge_attributes << ";\n";
        }
    }
  };

  // Produces commands to enclose in a tikzpicture in a LateX document:
  // ```
  // \begin{tikzpicture}
  //   <paste tikz commands here>
  // \end{tikzpicture}
  // ```
  // NB ensure TikZ styles have been defined.
  struct TikZVisitor
  {
    std::ostream &_os;

    template <typename Tree>
    void visit(Node const *node, Tree const &tree) const
    {
      auto const node_label = getNodeLabel(node, tree);
      auto const node_attributes = getNodeAttributes(node, tree);
      auto const bounding_volume = TreeAccess::getBoundingVolume(node, tree);
      auto const min_corner = bounding_volume.minCorner();
      auto const max_corner = bounding_volume.maxCorner();
      _os << R"(\draw)" << node_attributes << " " << min_corner << " rectangle "
          << max_corner << " node {" << node_label << "};\n";
    }
  };

  template <typename Tree, typename Visitor>
  static void visitAllIterative(Tree const &tree, Visitor const &visitor)
  {
    Stack<Node const *> stack;
    stack.emplace(TreeAccess::getRoot(tree));
    while (!stack.empty())
    {
      Node const *node = stack.top();
      stack.pop();

      visitor.visit(node, tree);

      if (!node->isLeaf())
        for (Node const *child : {node->children.first, node->children.second})
          stack.push(child);
    }
  }

  template <typename Tree, typename Predicate, typename Visitor>
  static int visit(Tree const &tree, Predicate const &pred,
                   Visitor const &visitor)
  {
    // The preprocessor directives below are intended to silent the
    // warnings about calling a __host__ function from a __host__
    // __device__ function emitted by nvcc.
#if defined(__CUDA_ARCH__)
    (void)tree;
    (void)pred;
    (void)visitor;
    throw std::runtime_error("not meant to execute on the GPU");
#else
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    Kokkos::View<Kokkos::pair<int, double> *, DeviceType> buffer("buffer", k);
    int const count = TreeTraversal<DeviceType>::nearestQuery(
        tree,
        [geometry, &visitor, &tree](Node const *node) {
          visitor.visit(node, tree);
          return distance(geometry, TreeAccess::getBoundingVolume(node, tree));
        },
        k, [](int, double) {}, buffer);
    return count;
#endif
  }
};
} // namespace Details
} // namespace ArborX

#endif
