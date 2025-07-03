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

#ifndef ARBORX_EXPAND_HALF_TO_FULL_HPP
#define ARBORX_EXPAND_HALF_TO_FULL_HPP

#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <class ExecutionSpace, class Offsets, class Indices>
void expandHalfToFull(ExecutionSpace const &space, Offsets &offsets,
                      Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::HalfToFull");
  typename Offsets::const_type const offsets_orig = offsets;
  typename Indices::const_type const indices_orig = indices;

  auto const n = offsets.extent(0) - 1;
  offsets = KokkosExt::cloneWithoutInitializingNorCopying(space, offsets_orig);
  Kokkos::deep_copy(space, offsets, 0);
  Kokkos::parallel_for(
      "ArborX::Experimental::HalfToFull::count",
      Kokkos::RangePolicy(space, 0, n), KOKKOS_LAMBDA(int i) {
        for (int j = offsets_orig(i); j < offsets_orig(i + 1); ++j)
        {
          int const k = indices_orig(j);
          Kokkos::atomic_inc(&offsets(i));
          Kokkos::atomic_inc(&offsets(k));
        }
      });
  KokkosExt::exclusive_scan(space, offsets, offsets, 0);

  auto const m = KokkosExt::lastElement(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices, m);

  auto counts = KokkosExt::clone(space, offsets,
                                 "ArborX::Experimental::HalfToFull::counts");
  Kokkos::parallel_for(
      "ArborX::Experimental::HalfToFull::rewrite",
      Kokkos::TeamPolicy(space, n, Kokkos::AUTO, 1),
      KOKKOS_LAMBDA(
          typename Kokkos::TeamPolicy<ExecutionSpace>::member_type const
              &member) {
        auto const i = member.league_rank();
        auto const first = offsets_orig(i);
        auto const last = offsets_orig(i + 1);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, last - first), [&](int j) {
              int const k = indices_orig(first + j);
              indices(Kokkos::atomic_fetch_inc(&counts(i))) = k;
              indices(Kokkos::atomic_fetch_inc(&counts(k))) = i;
            });
      });
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Details

#endif
