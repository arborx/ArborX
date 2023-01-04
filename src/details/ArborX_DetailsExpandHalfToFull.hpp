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

#ifndef ARBORX_DETAILS_EXPAND_HALF_TO_FULL_HPP
#define ARBORX_DETAILS_EXPAND_HALF_TO_FULL_HPP

#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsUtils.hpp>

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
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        for (int j = offsets_orig(i); j < offsets_orig(i + 1); ++j)
        {
          int const k = indices_orig(j);
          Kokkos::atomic_increment(&offsets(i));
          Kokkos::atomic_increment(&offsets(k));
        }
      });
  exclusivePrefixSum(space, offsets);

  auto const m = KokkosExt::lastElement(space, offsets);
  KokkosExt::reallocWithoutInitializing(space, indices, m);

  Offsets counts(
      Kokkos::view_alloc(space, "ArborX::Experimental::HalfToFull::count"), n);
  Kokkos::parallel_for(
      "ArborX::Experimental::HalfToFull::rewrite",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto const offsets_i = offsets(i);
        for (int j = offsets_orig(i); j < offsets_orig(i + 1); ++j)
        {
          int const k = indices_orig(j);
          indices(offsets_i + Kokkos::atomic_fetch_inc(&counts(i))) = k;
          indices(offsets(k) + Kokkos::atomic_fetch_inc(&counts(k))) = i;
        }
      });
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Details

#endif
