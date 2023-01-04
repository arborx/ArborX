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

#ifndef ARBORX_NEIGHBOR_LIST_HPP
#define ARBORX_NEIGHBOR_LIST_HPP

#include <ArborX_DetailsHalfTraversal.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp> // reallocWithoutInitializing
#include <ArborX_DetailsUtils.hpp>                // exclusivePrefixSum
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Experimental
{

struct NeighborListPredicateGetter
{
  float _radius;

  KOKKOS_FUNCTION auto operator()(Box b) const
  {
    return intersects(Sphere{b.minCorner(), _radius});
  }
};

template <class ExecutionSpace, class Primitives, class Offsets, class Indices>
void findHalfNeighborList(ExecutionSpace const &space,
                          Primitives const &primitives, float radius,
                          Offsets &offsets, Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList");

  using Details::HalfTraversal;

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::HalfNeighborList::Count");

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int, int j) { Kokkos::atomic_increment(&offsets(j)); },
      NeighborListPredicateGetter{radius});
  exclusivePrefixSum(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList::Fill");

  auto counts =
      KokkosExt::clone(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                       "ArborX::Experimental::HalfNeighborList::counts");
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        indices(Kokkos::atomic_fetch_inc(&counts(j))) = i;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

template <class ExecutionSpace, class Primitives, class Offsets, class Indices>
void findFullNeighborList(ExecutionSpace const &space,
                          Primitives const &primitives, float radius,
                          Offsets &offsets, Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList");

  using Details::HalfTraversal;

  using MemorySpace =
      typename AccessTraits<Primitives, PrimitivesTag>::memory_space;
  BVH<MemorySpace> bvh(space, primitives);
  int const n = bvh.size();

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::FullNeighborList::Count");

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        Kokkos::atomic_increment(&offsets(i));
        Kokkos::atomic_increment(&offsets(j));
      },
      NeighborListPredicateGetter{radius});
  exclusivePrefixSum(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList::Fill");

  auto counts =
      KokkosExt::clone(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                       "ArborX::Experimental::FullNeighborList::counts");
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        indices(Kokkos::atomic_fetch_inc(&counts(j))) = i;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList::Copy");

  auto counts_copy = KokkosExt::clone(space, counts, counts.label() + "_copy");
  Kokkos::parallel_for(
      "ArborX::Experimental::FullNeighborList::Copy",
      Kokkos::TeamPolicy<ExecutionSpace>(space, n, Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(
          typename Kokkos::TeamPolicy<ExecutionSpace>::member_type const
              &member) {
        auto const i = member.league_rank();
        auto const first = offsets(i);
        auto const last = counts_copy(i);
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, last - first), [&](int j) {
              int const k = indices(first + j);
              indices(Kokkos::atomic_fetch_inc(&counts(k))) = i;
            });
      });

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Experimental

#endif
