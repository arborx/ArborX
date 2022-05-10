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

#ifndef ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP
#define ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP

#include <ArborX_DetailsNode.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <utility> // declval

namespace ArborX
{
namespace Details
{
struct HappyTreeFriends
{
  template <class BVH>
  struct has_node_with_two_children
      : std::is_same<typename BVH::node_type::Tag, NodeWithTwoChildrenTag>::type
  {
  };

  template <class BVH>
  struct has_node_with_left_child_and_rope
      : std::is_same<typename BVH::node_type::Tag,
                     NodeWithLeftChildAndRopeTag>::type
  {
  };

  template <class BVH>
  static KOKKOS_FUNCTION int getRoot(BVH const &)
  {
    return 0;
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
    assert(isLeaf(bvh, i));
    const int n = bvh._internal_nodes.size();
    return bvh._leaf_nodes(i - n).bounding_volume;
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
    assert(!isLeaf(bvh, i));
    return bvh._internal_nodes(i).bounding_volume;
  }

  template <class BVH>
  static KOKKOS_FUNCTION bool isLeaf(BVH const &bvh, int i)
  {
    const int n = bvh._internal_nodes.size();
    return i >= n && bvh._leaf_nodes(i - n).isLeaf();
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeafPermutationIndex(BVH const &bvh, int i)
  {
    assert(isLeaf(bvh, i));
    const int n = bvh._internal_nodes.size();
    return bvh._leaf_nodes(i - n).getLeafPermutationIndex();
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeftChild(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    const int n = bvh._internal_nodes.size();
    if (i < n)
      return bvh._internal_nodes(i).left_child;
    else
      return bvh._leaf_nodes(i - n).left_child;
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getRightChild(BVH const &bvh, int i)
  {
#ifndef KOKKOS_ENABLE_CXX17
    return getRightChildImpl(bvh, i);
#else
    static_assert(has_node_with_two_children<BVH>::value ||
                  has_node_with_left_child_and_rope<BVH>::value);
    assert(!isLeaf(bvh, i));
    const int n = bvh._internal_nodes.size();
    if (i < n)
    {
      if constexpr (has_node_with_left_child_and_rope<BVH>::value)
        return bvh._internal_nodes(getLeftChild(bvh, i)).rope;
      else
        return bvh._internal_nodes(i).right_child;
    }
    else
    {
      if constexpr (has_node_with_left_child_and_rope<BVH>::value)
        return bvh._leaf_nodes(getLeftChild(bvh, i - n)).rope;
      else
        return bvh._leaf_nodes(i - n).right_child;
    }
#endif
  }

#ifndef KOKKOS_ENABLE_CXX17
  template <class BVH, std::enable_if_t<has_node_with_two_children<BVH>::value>
                           * = nullptr>
  static KOKKOS_FUNCTION auto getRightChildImpl(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    const int n = bvh._internal_nodes.size();
    if (i < n)
      return bvh._internal_nodes(i).right_child;
    else
      return bvh._leaf_nodes(i - n).right_child;
  }

  template <class BVH,
            std::enable_if_t<has_node_with_left_child_and_rope<BVH>::value> * =
                nullptr>
  static KOKKOS_FUNCTION auto getRightChildImpl(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    const int n = bvh._internal_nodes.size();
    if (i < n)
      return bvh._internal_nodes(getLeftChild(bvh, i)).rope;
    else
      return bvh._leaf_nodes(getLeftChild(bvh, i - n)).rope;
  }
#endif

  template <class BVH,
            std::enable_if_t<has_node_with_left_child_and_rope<BVH>::value> * =
                nullptr>
  static KOKKOS_FUNCTION auto getRope(BVH const &bvh, int i)
  {
    const int n = bvh._internal_nodes.size();
    if (i < n)
      return bvh._internal_nodes(i).rope;
    else
      return bvh._leaf_nodes(i - n).rope;
  }

  template <class BVH>
  static decltype(auto) getLeafNodes(BVH const &bvh)
  {
    return bvh.getLeafNodes();
  }
};
} // namespace Details
} // namespace ArborX

#endif
