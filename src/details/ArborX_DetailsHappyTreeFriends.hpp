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

  template <class BVH>
  static KOKKOS_FUNCTION
// FIXME_HIP See https://github.com/arborx/ArborX/issues/553
#ifdef __HIP_DEVICE_COMPILE__
      auto
#else
      auto const &
#endif
      getInternalBoundingVolume(BVH const &bvh, int i)
  {
    return bvh._internal_nodes(internalIndex(bvh, i)).bounding_volume;
  }

  template <class BVH>
  static KOKKOS_FUNCTION
// FIXME_HIP See https://github.com/arborx/ArborX/issues/553
#ifdef __HIP_DEVICE_COMPILE__
      auto
#else
      auto const &
#endif
      getLeafBoundingVolume(BVH const &bvh, int i)
  {
    static_assert(
        std::is_same_v<decltype(bvh._internal_nodes(0).bounding_volume),
                       decltype(bvh._leaf_nodes(0).value.bounding_volume)>);
    return bvh._leaf_nodes(i).value.bounding_volume;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto const &getValue(BVH const &bvh, int i)
  {
    assert(i >= 0 && i < (int)bvh.size());
    return bvh._leaf_nodes(i).value;
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
    return getRope(bvh, getLeftChild(bvh, i));
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRope(BVH const &bvh, int i)
  {
    return (isLeaf(bvh, i) ? bvh._leaf_nodes(i).rope
                           : bvh._internal_nodes(internalIndex(bvh, i)).rope);
  }
};
} // namespace ArborX::Details

#endif
