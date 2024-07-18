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

#ifndef ARBORX_DETAILS_UTILS_HPP
#define ARBORX_DETAILS_UTILS_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <typename ExecutionSpace, typename View, typename Offset>
void computeOffsetsInOrderedView(ExecutionSpace const &exec_space, View view,
                                 Offset &offsets)
{
  static_assert(KokkosExt::is_accessible_from<typename View::memory_space,
                                              ExecutionSpace>::value);
  static_assert(KokkosExt::is_accessible_from<typename Offset::memory_space,
                                              ExecutionSpace>::value);

  auto const n = view.extent_int(0);

  int num_offsets;
  KokkosExt::reallocWithoutInitializing(exec_space, offsets, n + 1);
  Kokkos::parallel_scan(
      "ArborX::Algorithms::compute_offsets_in_sorted_view",
      Kokkos::RangePolicy(exec_space, 0, n + 1),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cell_first_index =
            (i == 0 || i == n || view(i) != view(i - 1));
        if (is_cell_first_index)
        {
          if (final_pass)
            offsets(update) = i;
          ++update;
        }
      },
      num_offsets);
  Kokkos::resize(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing),
                 offsets, num_offsets);
}

} // namespace ArborX::Details

#endif
