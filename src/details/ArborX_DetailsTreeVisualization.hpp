/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
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

#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsTreeTraversal.hpp>

#include <Kokkos_Core.hpp>

#include <tuple> // ignore

namespace ArborX
{

namespace Details
{
std::ostream &operator<<(std::ostream &os, Point const &p)
{
  os << "(" << p[0] << "," << p[1] << ")";
  return os;
}

struct TreeVisualization
{
  template <typename Tree>
  static std::string getNodeLabel(Tree const &tree, int node)
  {
    auto const node_is_leaf = HappyTreeFriends::isLeaf(tree, node);
    auto const node_index =
        node_is_leaf ? HappyTreeFriends::getValue(tree, node).index : node;
    std::string label = node_is_leaf ? "l" : "i";
    label.append(std::to_string(node_index));
    return label;
  }

  template <typename Tree>
  static std::string getNodeAttributes(Tree const &tree, int node)
  {
    return HappyTreeFriends::isLeaf(tree, node) ? "[leaf]" : "[internal]";
  }

  template <typename Tree>
  static std::string getEdgeAttributes(Tree const &tree, int /*parent*/,
                                       int child)
  {
    return HappyTreeFriends::isLeaf(tree, child) ? "[pendant]" : "[edge]";
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
    void visit(Tree const &tree, int node) const
    {
      visitNode(tree, node);
      visitEdgesStartingFromNode(tree, node);
    }

    template <typename Tree>
    void visitNode(Tree const &tree, int node) const
    {
      auto const node_label = getNodeLabel(tree, node);
      auto const node_attributes = getNodeAttributes(tree, node);

      _os << "    " << node_label << " " << node_attributes << ";\n";
    }

    template <typename Tree>
    void visitEdgesStartingFromNode(Tree const &tree, int node) const
    {
      auto const node_label = getNodeLabel(tree, node);
      auto const node_is_internal = !HappyTreeFriends::isLeaf(tree, node);

      if (node_is_internal)
        for (auto const child : {HappyTreeFriends::getLeftChild(tree, node),
                                 HappyTreeFriends::getRightChild(tree, node)})
        {
          auto const child_label = getNodeLabel(tree, child);
          auto const edge_attributes = getEdgeAttributes(tree, node, child);

          _os << "    " << node_label << " -> " << child_label << " "
              << edge_attributes << ";\n";
        }
    }
  };

  // Produce commands to enclose in a tikzpicture in a LateX document:
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
    void visit(Tree const &tree, int node) const
    {
      auto const node_label = getNodeLabel(tree, node);
      auto const node_attributes = getNodeAttributes(tree, node);
      auto const bounding_volume =
          HappyTreeFriends::isLeaf(tree, node)
              ? HappyTreeFriends::getLeafBoundingVolume(tree, node)
              : HappyTreeFriends::getInternalBoundingVolume(tree, node);
      auto const min_corner = bounding_volume.minCorner();
      auto const max_corner = bounding_volume.maxCorner();
      _os << R"(\draw)" << node_attributes << " " << min_corner << " rectangle "
          << max_corner << " node {" << node_label << "};\n";
    }
  };

  template <typename Tree, typename Visitor>
  static void visitAllIterative(Tree const &tree, Visitor const &visitor)
  {
    Stack<int> stack;
    stack.emplace(HappyTreeFriends::getRoot(tree));
    while (!stack.empty())
    {
      auto const node = stack.top();
      stack.pop();

      visitor.visit(tree, node);

      if (!HappyTreeFriends::isLeaf(tree, node))
        for (auto const child : {HappyTreeFriends::getLeftChild(tree, node),
                                 HappyTreeFriends::getRightChild(tree, node)})
          stack.push(child);
    }
  }

  template <typename TreeType, typename VisitorType, typename Permute>
  struct VisitorCallback
  {
    template <typename Query>
    KOKKOS_FUNCTION void
    operator()(Query const &, typename TreeType::value_type const &value) const
    {
      _visitor.visit(_tree, permute(value.index));
    }

    TreeType _tree;
    VisitorType _visitor;
    Permute permute;
  };

  template <typename Tree, typename Predicate, typename Visitor>
  static void visit(Tree const &tree, Predicate const &pred,
                    Visitor const &visitor)
  {
    // The preprocessor directives below are intended to silence the
    // warnings about calling a __host__ function from a __host__
    // __device__ function emitted by nvcc.
#if defined(__CUDA_ARCH__)
    std::ignore = tree;
    std::ignore = pred;
    std::ignore = visitor;
    throw std::runtime_error("visit() is not meant to execute on the GPU");
#else
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using Predicates = Kokkos::View<Predicate *, ExecutionSpace>;
    using Permute = Kokkos::View<int *, ExecutionSpace>;
    using Callback = VisitorCallback<Tree, Visitor, Permute>;

    ExecutionSpace space;

    int const n = tree.size();
    Permute permute(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                       "ArborX::permute"),
                    n);
    Kokkos::parallel_for(
        "ArborX::Viz::compute_permutation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
          permute(HappyTreeFriends::getValue(tree, i).index) = i;
        });

    Predicates predicates(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                             "ArborX::predicates"),
                          1);
    predicates(0) = pred;

    TreeTraversal<Tree, Predicates, Callback, NearestPredicateTag>
        tree_traversal(ExecutionSpace{}, tree, predicates,
                       Callback{tree, visitor, permute});
#endif
  }
};
} // namespace Details
} // namespace ArborX

#endif
