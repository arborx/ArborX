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

#ifndef ARBORX_NODE_HPP
#define ARBORX_NODE_HPP

#include <ArborX_Box.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <utility> // std::move

namespace ArborX::Details
{

constexpr int ROPE_SENTINEL = -1;

template <class BoundingVolume>
struct PairIndexVolume
{
  unsigned index;
  BoundingVolume bounding_volume;
};

template <class Value>
struct LeafNode
{
  using value_type = Value;

  int rope = ROPE_SENTINEL;
  Value value;
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

template <class Value>
KOKKOS_INLINE_FUNCTION constexpr LeafNode<Value>
makeLeafNode(Value value) noexcept
{
  return {ROPE_SENTINEL, std::move(value)};
}

} // namespace ArborX::Details

#endif
