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

#ifndef ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP
#define ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP

#include <ArborX_DetailsNode.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <utility> // declval

namespace ArborX::Details
{
struct LeafNodeTag
{};
struct InternalNodeTag
{};

struct HappyTreeFriends
{
  template <class BVH>
  static KOKKOS_FUNCTION int getRoot(BVH const &bvh)
  {
    assert(bvh.size() > 1);
    return bvh.size();
  }

  template <class BVH>
  static KOKKOS_FUNCTION bool isLeaf(BVH const &bvh, int i)
  {
    assert(bvh.size() > 1);
    assert(i >= 0 && i < 2 * (int)bvh.size() - 1);
    return i < (int)bvh.size();
  }

  template <class BVH>
  static KOKKOS_FUNCTION int internalIndex(BVH const &bvh, int i)
  {
    return i - (int)bvh.size();
  }

  template <class Tag, class BVH>
  static KOKKOS_FUNCTION
// FIXME_HIP See https://github.com/arborx/ArborX/issues/553
#ifdef __HIP_DEVICE_COMPILE__
      auto
#else
      auto const &
#endif
      getBoundingVolume(Tag, BVH const &bvh, int i)
  {
    static_assert(
        std::is_same_v<decltype(bvh._internal_nodes(0).bounding_volume),
                       decltype(bvh._leaf_nodes(0).bounding_volume)>);
    if constexpr (std::is_same_v<Tag, InternalNodeTag>)
      return bvh._internal_nodes(i).bounding_volume;
    else
      return bvh._leaf_nodes(i).bounding_volume;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeafPermutationIndex(BVH const &bvh, int i)
  {
    assert(i >= 0 && i < (int)bvh.size());
    return bvh._leaf_nodes(i).permutation_index;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeftChild(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    return bvh._internal_nodes(internalIndex(bvh, i)).left_child;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRightChild(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    auto left_child = getLeftChild(bvh, i);
    bool const is_leaf = isLeaf(bvh, left_child);
    return (is_leaf ? getRope(LeafNodeTag{}, bvh, left_child)
                    : getRope(InternalNodeTag{}, bvh,
                              internalIndex(bvh, left_child)));
  }

  template <class Tag, class BVH>
  static KOKKOS_FUNCTION auto getRope(Tag, BVH const &bvh, int i)
  {
    if constexpr (std::is_same_v<Tag, InternalNodeTag>)
      return bvh._internal_nodes(i).rope;
    else
      return bvh._leaf_nodes(i).rope;
  }
};
} // namespace ArborX::Details

#endif
