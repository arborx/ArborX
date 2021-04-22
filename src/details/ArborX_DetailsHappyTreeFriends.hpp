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

#ifndef ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP
#define ARBORX_DETAILS_HAPPY_TREE_FRIENDS_HPP

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
  static KOKKOS_FUNCTION decltype(auto) getNodePtr(BVH const &bvh, int i)
  {
    return bvh.getNodePtr(i);
  }

  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto) getRoot(BVH const &bvh)
  {
    return bvh.getRoot();
  }

  template <class BVH>
  using node_const_pointer_t = decltype(getRoot(std::declval<BVH const &>()));

  template <class BVH>
  using node_t =
      std::remove_const_t<std::remove_pointer_t<node_const_pointer_t<BVH>>>;

  template <class BVH>
  static KOKKOS_FUNCTION decltype(auto)
  getBoundingVolume(BVH const &bvh, node_t<BVH> const *node)
  {
    return bvh.getBoundingVolume(node);
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
