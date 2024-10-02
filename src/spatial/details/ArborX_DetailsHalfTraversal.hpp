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

#ifndef ARBORX_DETAILS_HALF_TRAVERSAL_HPP
#define ARBORX_DETAILS_HALF_TRAVERSAL_HPP

#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsNode.hpp> // ROPE_SENTINEL

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <class BVH, class Callback, class PredicateGetter>
struct HalfTraversal
{
  BVH _bvh;
  PredicateGetter _get_predicate;
  Callback _callback;

  template <class ExecutionSpace>
  HalfTraversal(ExecutionSpace const &space, BVH const &bvh,
                Callback const &callback, PredicateGetter const &getter)
      : _bvh{bvh}
      , _get_predicate{getter}
      , _callback{callback}
  {
    if (_bvh.empty())
    {
      // do nothing
    }
    else if (_bvh.size() == 1)
    {
      // do nothing either
    }
    else
    {
      Kokkos::parallel_for("ArborX::Experimental::HalfTraversal",
                           Kokkos::RangePolicy(space, 0, _bvh.size()), *this);
    }
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const leaf_value = HappyTreeFriends::getValue(_bvh, i);
    auto const predicate = _get_predicate(leaf_value);

    int node = HappyTreeFriends::getRope(_bvh, i);
    while (node != ROPE_SENTINEL)
    {
      if (HappyTreeFriends::isLeaf(_bvh, node))
      {
        if (predicate(HappyTreeFriends::getIndexable(_bvh, node)))
          _callback(leaf_value, HappyTreeFriends::getValue(_bvh, node));
        node = HappyTreeFriends::getRope(_bvh, node);
      }
      else
      {
        node =
            (predicate(HappyTreeFriends::getInternalBoundingVolume(_bvh, node))
                 ? HappyTreeFriends::getLeftChild(_bvh, node)
                 : HappyTreeFriends::getRope(_bvh, node));
      }
    }
  }
};

} // namespace ArborX::Details

#endif
