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

#ifndef ARBORX_DETAILS_SELF_COLLISION_HPP
#define ARBORX_DETAILS_SELF_COLLISION_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsHeap.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsNode.hpp> // ROPE_SENTINEL
#include <ArborX_DetailsOperatorFunctionObjects.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_Sphere.hpp>

namespace ArborX::Details
{

template <class BVH, class Predicate, class Callback>
struct SelfCollisionSpatial
{
  BVH _bvh;
  Predicate _predicate;
  Callback _callback;

  template <class ExecutionSpace>
  SelfCollisionSpatial(ExecutionSpace const &space, BVH const &bvh,
                       Predicate const &predicate, Callback const &callback)
      : _bvh{bvh}
      , _predicate{predicate}
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
          "ArborX::SelfCollision::spatial",
          Kokkos::RangePolicy<ExecutionSpace>(space, n - 1, 2 * n - 1), *this);
    }
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const predicate =
        _predicate(HappyTreeFriends::getBoundingVolume(_bvh, i));
    auto const leaf_permutation_i =
        HappyTreeFriends::getLeafPermutationIndex(_bvh, i);

    int node;
#ifdef V1
    int next = HappyTreeFriends::getRoot(_bvh); // start with root
#else
    int next = HappyTreeFriends::getRope(_bvh, i);
#endif
    do
    {
      node = next;

      if (predicate(HappyTreeFriends::getBoundingVolume(_bvh, node)))
      {
        if (!HappyTreeFriends::isLeaf(_bvh, node))
        {
          auto const left_child = HappyTreeFriends::getLeftChild(_bvh, node);
#ifdef V1
          if (left_child < i)
#else
          if (true)
#endif
          {
            next = left_child;
          }
          else
          {
            next = HappyTreeFriends::getRightChild(_bvh, node);
          }
        }
        else
        {
#ifdef V1
          if (node < i)
#else
          if (true)
#endif
          {
            _callback(leaf_permutation_i,
                      HappyTreeFriends::getLeafPermutationIndex(_bvh, node));
          }
          next = HappyTreeFriends::getRope(_bvh, node);
        }
      }
      else
      {
        next = HappyTreeFriends::getRope(_bvh, node);
      }

    } while (next != ROPE_SENTINEL);
  }
};

template <class ExecutionSpace, class Primitives, class Offsets, class Indices>
void cabana_proxy(ExecutionSpace const &space, Primitives const &primitives,
                  float radius, Offsets &offsets, Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::CabanaProxy");

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  auto const predicate = KOKKOS_LAMBDA(Box b)
  {
    return intersects(Sphere{b.minCorner(), radius});
  };

  Kokkos::Profiling::pushRegion("ArborX::Experimental::Count");

#define HALF_V2
  Kokkos::View<int *, MemorySpace> counts(
      Kokkos::view_alloc(space, "ArborX::counts"), n);
  SelfCollisionSpatial(
      space, bvh, predicate, KOKKOS_LAMBDA(int i, int j) {
#if defined(HALF_V1)
        ++counts(i);
        (void)j;
#elif defined(HALF_V2)
        Kokkos::atomic_increment(&counts(j));
        (void)i;
#elif defined(FULL_V1)
        Kokkos::atomic_increment(&counts(i));
        Kokkos::atomic_increment(&counts(j));
#else
#error faulty logic
#endif
      });

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                    counts);
  exclusivePrefixSum(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::Fill");

  Kokkos::deep_copy(space, counts, 0);
  SelfCollisionSpatial(
      space, bvh, predicate, KOKKOS_LAMBDA(int i, int j) {
#if defined(HALF_V1)
        indices(offsets(i) + counts(i)++) = j;
#elif defined(HALF_V2)
        indices(offsets(j) + Kokkos::atomic_fetch_inc(&counts(j))) = i;
#elif defined(FULL_V1)
        indices(offsets(i) + Kokkos::atomic_fetch_inc(&counts(i))) = j;
        indices(offsets(j) + Kokkos::atomic_fetch_inc(&counts(j))) = i;
#else
#error faulty logic
#endif
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Details

#endif
