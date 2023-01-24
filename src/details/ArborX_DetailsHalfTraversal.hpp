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
      auto const n = _bvh.size();
      Kokkos::parallel_for(
          "ArborX::Experimental::HalfTraversal",
          Kokkos::RangePolicy<ExecutionSpace>(space, n - 1, 2 * n - 1), *this);
    }
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const predicate =
        _get_predicate(HappyTreeFriends::getBoundingVolume(_bvh, i));
    auto const leaf_permutation_i =
        HappyTreeFriends::getLeafPermutationIndex(_bvh, i);

    int node = HappyTreeFriends::getRope(_bvh, i);
    while (node != ROPE_SENTINEL)
    {
      if (predicate(HappyTreeFriends::getBoundingVolume(_bvh, node)))
      {
        if (!HappyTreeFriends::isLeaf(_bvh, node))
        {
          node = HappyTreeFriends::getLeftChild(_bvh, node);
        }
        else
        {
          _callback(leaf_permutation_i,
                    HappyTreeFriends::getLeafPermutationIndex(_bvh, node));
          node = HappyTreeFriends::getRope(_bvh, node);
        }
      }
      else
      {
        node = HappyTreeFriends::getRope(_bvh, node);
      }
    }
  }
};

} // namespace ArborX::Details

#endif
