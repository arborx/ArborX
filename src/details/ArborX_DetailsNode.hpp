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

#ifndef ARBORX_NODE_HPP
#define ARBORX_NODE_HPP

#include <ArborX_Box.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <climits> // INT_MIN
#include <utility> // std::move

namespace ArborX
{
namespace Details
{

constexpr int ROPE_SENTINEL = -1;

template <class BoundingVolume>
struct NodeWithLeftChildAndRope
{
  using bounding_volume_type = BoundingVolume;

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

  KOKKOS_INLINE_FUNCTION constexpr unsigned
  getLeafPermutationIndex() const noexcept
  {
    assert(isLeaf());
    return -left_child;
  }

  // The meaning of the left child depends on whether the node is an internal
  // node, or a leaf. For internal nodes, it is the child from the left
  // subtree. For a leaf node, it is the negative of the permutation index.
  int left_child = INT_MIN;

  // An interesting property to remember: a right child is always the rope of
  // the left child.
  int rope = ROPE_SENTINEL;

  BoundingVolume bounding_volume;
};

template <class BoundingVolume>
KOKKOS_INLINE_FUNCTION constexpr NodeWithLeftChildAndRope<BoundingVolume>
makeLeafNode(unsigned permutation_index,
             BoundingVolume bounding_volume) noexcept
{
  return {-static_cast<int>(permutation_index), ROPE_SENTINEL,
          std::move(bounding_volume)};
}
} // namespace Details
} // namespace ArborX

#endif
