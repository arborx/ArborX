/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_TREE_NODE_LABELING_HPP
#define ARBORX_DETAILS_TREE_NODE_LABELING_HPP

#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

template <class ExecutionSpace, class BVH, class Parents>
void findParents(ExecutionSpace const &exec_space, BVH const &bvh,
                 Parents const &parents)
{
  int const n = bvh.size();

  ARBORX_ASSERT(n >= 2);
  ARBORX_ASSERT((int)parents.size() == 2 * n - 1);

  Kokkos::parallel_for(
      "ArborX::recompute_internal_and_leaf_node_parents",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n - 1),
      KOKKOS_LAMBDA(int i) {
        parents(HappyTreeFriends::getLeftChild(bvh, i)) = i;
        parents(HappyTreeFriends::getRightChild(bvh, i)) = i;
      });
}

template <class ExecutionSpace, class Parents, class Labels>
void reduceLabels(ExecutionSpace const &exec_space, Parents const &parents,
                  Labels labels)
{
  int const n = (parents.size() + 1) / 2;

  ARBORX_ASSERT(n >= 2);
  ARBORX_ASSERT(labels.size() == parents.size());

  using ValueType = typename Labels::value_type;
  constexpr ValueType indeterminate = -1;
  constexpr ValueType untouched = -2;

  // Reset parent labels
  auto internal_node_labels =
      Kokkos::subview(labels, std::make_pair(0, (int)n - 1));
  Kokkos::deep_copy(exec_space, internal_node_labels, untouched);
  Kokkos::parallel_for(
      "ArborX::reduce_internal_node_labels",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, n - 1, 2 * n - 1),
      KOKKOS_LAMBDA(int i) {
        assert(labels(i) != indeterminate);
        assert(labels(i) != untouched);
        assert(parents(i) >= 0);

        // TODO consider asserting the precondition below holds at call site or
        // taking root as an input argument
        constexpr int root = 0; // Details::HappyTreeFriends::getRoot(bvh)
        do
        {
          int const label = labels(i);
          int const parent = parents(i);

          int const parent_label = Kokkos::atomic_compare_exchange(
              &labels(parent), untouched, label);

          // Terminate the first thread and let the second one continue. This
          // ensures that every node is processed only once, and only after
          // processing both of its children.
          if (parent_label == untouched)
            break;

          // Set the parent to indeterminate if children's labels do not match
          if (parent_label != label)
            labels(parent) = indeterminate;

          i = parent;
        } while (i != root);
      });
}

} // namespace Details
} // namespace ArborX

#endif
