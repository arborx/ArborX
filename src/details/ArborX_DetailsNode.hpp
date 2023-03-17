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
#include <climits> // UINT_MAX
#include <utility> // std::move

namespace ArborX
{
namespace Details
{

constexpr int ROPE_SENTINEL = -1;

template <class BoundingVolume>
struct LeafNode
{
  using bounding_volume_type = BoundingVolume;

  unsigned permutation_index = UINT_MAX;
  int rope = ROPE_SENTINEL;
  BoundingVolume bounding_volume;
};

template <class BoundingVolume>
struct InternalNode
{
  using bounding_volume_type = BoundingVolume;

  // Right child is the rope of the left child
  int left_child = -1;
  int rope = ROPE_SENTINEL;
  BoundingVolume bounding_volume;
};

template <class BoundingVolume>
KOKKOS_INLINE_FUNCTION constexpr LeafNode<BoundingVolume>
makeLeafNode(unsigned permutation_index,
             BoundingVolume bounding_volume) noexcept
{
  return {permutation_index, ROPE_SENTINEL, std::move(bounding_volume)};
}
} // namespace Details
} // namespace ArborX

#endif
