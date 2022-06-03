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
  {};

  template <class BVH>
  struct has_node_with_left_child_and_rope
      : std::is_same<typename BVH::node_type::Tag,
                     NodeWithLeftChildAndRopeTag>::type
  {};

  template <class BVH>
  static KOKKOS_FUNCTION int getRoot(BVH const &)
  {
    return 0;
  }

  template <class BVH>
// FIXME_HIP See https://github.com/arborx/ArborX/issues/553
#ifdef __HIP_DEVICE_COMPILE__
  static KOKKOS_FUNCTION auto getBoundingVolume(BVH const &bvh, int i)
#else
  static KOKKOS_FUNCTION auto const &getBoundingVolume(BVH const &bvh, int i)
#endif
  {
    return bvh._internal_and_leaf_nodes(i).bounding_volume;
  }

  template <class BVH>
  static KOKKOS_FUNCTION bool isLeaf(BVH const &bvh, int i)
  {
    return bvh._internal_and_leaf_nodes(i).isLeaf();
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeafPermutationIndex(BVH const &bvh, int i)
  {
    assert(isLeaf(bvh, i));
    return bvh._internal_and_leaf_nodes(i).getLeafPermutationIndex();
  }

  template <class BVH>
  static KOKKOS_FUNCTION auto getLeftChild(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    return bvh._internal_and_leaf_nodes(i).left_child;
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
    if constexpr (has_node_with_left_child_and_rope<BVH>::value)
      return bvh._internal_and_leaf_nodes(getLeftChild(bvh, i)).rope;
    else
      return bvh._internal_and_leaf_nodes(i).right_child;
#endif
  }

#ifndef KOKKOS_ENABLE_CXX17
  template <class BVH, std::enable_if_t<has_node_with_two_children<BVH>::value>
                           * = nullptr>
  static KOKKOS_FUNCTION auto getRightChildImpl(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    return bvh._internal_and_leaf_nodes(i).right_child;
  }

  template <class BVH,
            std::enable_if_t<has_node_with_left_child_and_rope<BVH>::value> * =
                nullptr>
  static KOKKOS_FUNCTION auto getRightChildImpl(BVH const &bvh, int i)
  {
    assert(!isLeaf(bvh, i));
    return bvh._internal_and_leaf_nodes(getLeftChild(bvh, i)).rope;
  }
#endif

  template <class BVH,
            std::enable_if_t<has_node_with_left_child_and_rope<BVH>::value> * =
                nullptr>
  static KOKKOS_FUNCTION auto getRope(BVH const &bvh, int i)
  {
    return bvh._internal_and_leaf_nodes(i).rope;
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
