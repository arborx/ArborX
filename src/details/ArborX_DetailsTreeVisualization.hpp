/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

template <typename DeviceType>
struct TreeVisualization
{
  static_assert(KokkosExt::is_accessible_from<typename DeviceType::memory_space,
                                              Kokkos::HostSpace>{},
                "tree visualization only available on the host");
  struct TreeAccess
  {
    template <typename Tree>
    KOKKOS_INLINE_FUNCTION static typename Tree::node_type const *
    getLeaf(Tree const &tree, size_t index)
    {
      auto leaf_nodes = tree.getLeafNodes();
      auto const *first = leaf_nodes.data();
      auto const *last = first + static_cast<ptrdiff_t>(leaf_nodes.size());
      for (; first != last; ++first)
        if (index == first->getLeafPermutationIndex())
          return first;
      return nullptr;
    }

    template <typename Tree>
    KOKKOS_INLINE_FUNCTION static int
    getIndex(typename Tree::node_type const *node, Tree const &tree)
    {
      return node->isLeaf() ? node->getLeafPermutationIndex()
                            : node - tree.getRoot();
    }

    template <typename Tree>
    KOKKOS_INLINE_FUNCTION static typename Tree::node_type const *
    getRoot(Tree const &tree)
    {
      return tree.getRoot();
    }

    template <typename Tree>
    KOKKOS_INLINE_FUNCTION static typename Tree::node_type const *
    getNodePtr(Tree const &tree, int index)
    {
      return tree.getNodePtr(index);
    }

    template <typename Tree>
    KOKKOS_INLINE_FUNCTION static typename Tree::bounding_volume_type
    getBoundingVolume(typename Tree::node_type const *node, Tree const &tree)
    {
      return tree.getBoundingVolume(node);
    }
  };

  template <typename Tree>
  static std::string getNodeLabel(typename Tree::node_type const *node,
                                  Tree const &tree)
  {
    auto const node_is_leaf = node->isLeaf();
    auto const node_index = TreeAccess::getIndex(node, tree);
    std::string label = node_is_leaf ? "l" : "i";
    label.append(std::to_string(node_index));
    return label;
  }

  template <typename Tree>
  static std::string getNodeAttributes(typename Tree::node_type const *node,
                                       Tree const &)
  {
    return node->isLeaf() ? "[leaf]" : "[internal]";
  }

  template <typename Tree>
  static std::string
  getEdgeAttributes(typename Tree::node_type const * /*parent*/,
                    typename Tree::node_type const *child, Tree const &)
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
    void visit(typename Tree::node_type const *node, Tree const &tree) const
    {
      visitNode(node, tree);
      visitEdgesStartingFromNode(node, tree);
    }

    template <typename Node, typename Tree>
    void visitNode(Node const *node, Tree const &tree) const
    {
      auto const node_label = getNodeLabel(node, tree);
      auto const node_attributes = getNodeAttributes(node, tree);

      _os << "    " << node_label << " " << node_attributes << ";\n";
    }

    template <typename Tree>
    void visitEdgesStartingFromNode(typename Tree::node_type const *node,
                                    Tree const &tree) const
    {
      auto const node_label = getNodeLabel(node, tree);
      auto const node_is_internal = !node->isLeaf();
      auto getNodePtr = [&](int i) { return TreeAccess::getNodePtr(tree, i); };

      if (node_is_internal)
        for (auto const *child :
             {getNodePtr(node->left_child), getNodePtr(node->right_child)})
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
    void visit(typename Tree::node_type const *node, Tree const &tree) const
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
    using Node = typename Tree::node_type;
    Stack<Node const *> stack;
    stack.emplace(TreeAccess::getRoot(tree));
    auto getNodePtr = [&](int i) { return TreeAccess::getNodePtr(tree, i); };
    while (!stack.empty())
    {
      auto const *node = stack.top();
      stack.pop();

      visitor.visit(node, tree);

      if (!node->isLeaf())
        for (auto const *child :
             {getNodePtr(node->left_child), getNodePtr(node->right_child)})
          stack.push(child);
    }
  }

  template <typename TreeType, typename VisitorType>
  struct VisitorCallback
  {
    template <typename Query>
    KOKKOS_FUNCTION void operator()(Query const &, int index) const
    {
      _visitor.visit(_tree.getNodePtr(index), _tree);
    }

    TreeType _tree;
    VisitorType _visitor;
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
    using ExecutionSpace = typename DeviceType::execution_space;
    using Predicates = Kokkos::View<Predicate *, DeviceType>;
    using Callback = VisitorCallback<Tree, Visitor>;

    Predicates predicates("predicates", 1);
    predicates(0) = pred;

    TreeTraversal<Tree, Predicates, Callback, NearestPredicateTag>
        tree_traversal(ExecutionSpace{}, tree, predicates,
                       Callback{tree, visitor});
#endif
  }
};
} // namespace Details
} // namespace ArborX

#endif
