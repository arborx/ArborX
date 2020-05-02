/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_HALO_FINDER_HPP
#define ARBORX_HALO_FINDER_HPP

#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_View.hpp>

namespace ArborX
{

namespace HaloFinder
{

template <typename ExecutionSpace, typename... P>
void findHalos(ExecutionSpace exec_space, Kokkos::View<int *, P...> ccs,
               Kokkos::View<int *, P...> &halos_offset,
               Kokkos::View<int *, P...> &halos_indices, int min_size = 2)
{
  using MemorySpace = typename Kokkos::View<int *, P...>::memory_space;

  auto const num_nodes = ccs.extent_int(0);

  Kokkos::Profiling::pushRegion("ArborX:HaloFinder");

  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:sort_and_filter_ccs");

  // sort ccs and compute permutation
  auto permute = Details::sortObjects(exec_space, ccs);

  reallocWithoutInitializing(halos_offset, num_nodes + 1);
  Kokkos::View<int *, MemorySpace> halos_starts(
      Kokkos::ViewAllocateWithoutInitializing("halos_starts"), num_nodes);
  int num_halos = 0;
  // In the following scan, we locate the starting position (stored in
  // halos_starts) and size (stored in halos_offset) of each valid halo (i.e.,
  // connected component of size >= min_size). For every index i, we check
  // whether its CC index is different from the previous one (this indicates a
  // start of connected component) and whether the CC index of i + min_size is
  // the same (this indicates that this CC is at least of min_size size). If
  // those are true, we do a linear search from i + min_size till next CC index
  // change to find the CC size.
  Kokkos::parallel_scan(
      ARBORX_MARK_REGION("compute_halos_starts_and_sizes"),
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nodes),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        if ((i + min_size - 1 < num_nodes) &&
            (i == 0 || ccs(i) != ccs(i - 1)) &&
            (ccs(i + min_size - 1) == ccs(i)))
        {
          if (final_pass)
          {
            halos_starts(update) = i;
            int end = i + min_size - 1;
            while (++end < num_nodes && ccs(end) == ccs(i))
              ; // do nothing
            halos_offset(update) = end - i;
          }
          ++update;
        }
      },
      num_halos);
  Kokkos::resize(halos_offset, num_halos + 1);
  exclusivePrefixSum(exec_space, halos_offset);

  // Copy ccs indices to halos
  reallocWithoutInitializing(halos_indices, lastElement(halos_offset));
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("populate_halos"),
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        for (int k = halos_offset(i); k < halos_offset(i + 1); ++k)
        {
          halos_indices(k) = permute(halos_starts(i) + (k - halos_offset(i)));
        }
      });

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

} // namespace HaloFinder
} // namespace ArborX

#endif
