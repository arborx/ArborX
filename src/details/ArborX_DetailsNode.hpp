/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_NODE_HPP
#define ARBORX_NODE_HPP

#include <ArborX_Box.hpp>
#include <details/ArborX_Exception.hpp>

#include <Kokkos_Pair.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{

struct NodeWithTwoChildrenTag
{
};
struct NodeWithLeftChildAndRopeTag
{
};

struct NodeWithTwoChildren
{
  using Tag = NodeWithTwoChildrenTag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr NodeWithTwoChildren() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept
  {
    return left_child == -1;
  }

  KOKKOS_INLINE_FUNCTION constexpr std::size_t getLeafPermutationIndex() const
      noexcept
  {
    ARBORX_ASSERT(isLeaf());
    return right_child;
  }

  int left_child = -1;
  int right_child = -1;
  Box bounding_box;
};

KOKKOS_INLINE_FUNCTION constexpr NodeWithTwoChildren
makeLeafNode(NodeWithTwoChildrenTag, std::size_t permutation_index,
             Box box) noexcept
{
  return {-1, static_cast<int>(permutation_index), std::move(box)};
} // namespace Details

int constexpr ROPE_SENTINEL = -1;

struct NodeWithLeftChildAndRope
{
  using Tag = NodeWithLeftChildAndRopeTag;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr NodeWithLeftChildAndRope() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept
  {
    // Important: this check works only as long as the internal node with index
    // 0 is at the root. If there is a need in the future, this can be changed
    // to "< 0", but would require additional arithmetic (subtracting 1) in
    // `getLeafPermutationIndex` and in `makeLeafNode`.
    return left_child <= 0;
  }

  KOKKOS_INLINE_FUNCTION constexpr std::size_t getLeafPermutationIndex() const
      noexcept
  {
    ARBORX_ASSERT(isLeaf());
    return -left_child;
  }

  // The meaning of the left child depends on whether the node is an internal
  // node, or a leaf. For internal nodes, it is the child from the left
  // subtree. For a leaf node, it is the negative of the permutation index.
  int left_child = INT_MIN;

  // An interesting property to remember: a right child is always the rope of
  // the left child.
  int rope = ROPE_SENTINEL;

  Box bounding_box;
};

KOKKOS_INLINE_FUNCTION constexpr NodeWithLeftChildAndRope
makeLeafNode(NodeWithLeftChildAndRopeTag, std::size_t permutation_index,
             Box box) noexcept
{
  return {-static_cast<int>(permutation_index), ROPE_SENTINEL, std::move(box)};
}
} // namespace Details
} // namespace ArborX

#endif
