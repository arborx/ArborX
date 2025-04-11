/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
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

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <detail/ArborX_HalfTraversal.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp> // reallocWithoutInitializing

#include <Kokkos_Core.hpp>

namespace ArborX::Experimental
{

struct NeighborListPredicateGetter
{
  float _radius;

  template <typename Point, typename Index>
  KOKKOS_FUNCTION auto
  operator()(PairValueIndex<Point, Index> const &pair) const
  {
    static_assert(GeometryTraits::is_point_v<Point>);

    constexpr int dim = GeometryTraits::dimension_v<Point>;
    using Coordinate = typename GeometryTraits::coordinate_type_t<Point>;
    return intersects(
        Sphere{Details::convert<::ArborX::Point<dim, Coordinate>>(pair.value),
               _radius});
  }
};

template <class ExecutionSpace, class Primitives, class Offsets, class Indices>
void findHalfNeighborList(ExecutionSpace const &space,
                          Primitives const &primitives, float radius,
                          Offsets &offsets, Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList");

  namespace KokkosExt = ArborX::Details::KokkosExt;
  using Details::HalfTraversal;

  using Points = Details::AccessValues<Primitives>;

  using MemorySpace = typename Points::memory_space;
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);

  Points points{primitives}; // NOLINT
  int const n = points.size();

  using Value = PairValueIndex<Point>;

  BoundingVolumeHierarchy bvh(space, Experimental::attach_indices(points));

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::HalfNeighborList::Count");

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(Value const &, Value const &value) {
        Kokkos::atomic_inc(&offsets(value.index));
      },
      NeighborListPredicateGetter{radius});
  KokkosExt::exclusive_scan(space, offsets, offsets, 0);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfNeighborList::Fill");

  auto counts =
      KokkosExt::clone(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                       "ArborX::Experimental::HalfNeighborList::counts");
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(Value const &value1, Value const &value2) {
        indices(Kokkos::atomic_fetch_inc(&counts(value2.index))) = value1.index;
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

  namespace KokkosExt = ArborX::Details::KokkosExt;
  using Details::HalfTraversal;

  using Points = Details::AccessValues<Primitives>;

  using MemorySpace = typename Points::memory_space;
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);

  Points points{primitives}; // NOLINT
  int const n = points.size();

  using Value = PairValueIndex<Point>;

  BoundingVolumeHierarchy bvh(space, Experimental::attach_indices(points));

  Kokkos::Profiling::pushRegion(
      "ArborX::Experimental::FullNeighborList::Count");

  KokkosExt::reallocWithoutInitializing(space, offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(Value const &value1, Value const &value2) {
        Kokkos::atomic_inc(&offsets(value1.index));
        Kokkos::atomic_inc(&offsets(value2.index));
      },
      NeighborListPredicateGetter{radius});
  KokkosExt::exclusive_scan(space, offsets, offsets, 0);
  KokkosExt::reallocWithoutInitializing(space, indices,
                                        KokkosExt::lastElement(space, offsets));

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList::Fill");

  auto counts =
      KokkosExt::clone(space, Kokkos::subview(offsets, std::make_pair(0, n)),
                       "ArborX::Experimental::FullNeighborList::counts");
  HalfTraversal(
      space, bvh,
      KOKKOS_LAMBDA(Value const &value1, Value const &value2) {
        indices(Kokkos::atomic_fetch_inc(&counts(value2.index))) = value1.index;
      },
      NeighborListPredicateGetter{radius});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FullNeighborList::Copy");

  auto counts_copy = KokkosExt::clone(space, counts, counts.label() + "_copy");
  Kokkos::parallel_for(
      "ArborX::Experimental::FullNeighborList::Copy",
      Kokkos::TeamPolicy(space, n, Kokkos::AUTO, 1),
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
