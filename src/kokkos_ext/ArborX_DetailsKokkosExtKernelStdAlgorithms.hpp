/****************************************************************************
 * Copyright (c) 2017-2024 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_KOKKOS_EXT_KERNEL_STD_ALGORITHMS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_KERNEL_STD_ALGORITHMS_HPP

#include <ArborX_DetailsKokkosExtSwap.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::Details::KokkosExt
{

template <typename Iterator>
KOKKOS_FUNCTION void nth_element(Iterator first, Iterator nth, Iterator last)
{
  if (first == last || nth == last)
    return;

  // Lomuto partitioning
  auto partition = [](Iterator left, Iterator right, Iterator pivot) {
    using KokkosExt::swap;

    --right;

    swap(*pivot, *right);
    auto it_i = left;
    auto it_j = left;
    while (it_j < right)
    {
      if (*it_j < *right)
        swap(*it_j, *(it_i++));
      ++it_j;
    }
    swap(*it_i, *right);
    return it_i;
  };

  // Simple quickselect implementation
  while (true)
  {
    if (first == last)
      return;

    // Choosing nth element as a pivot should lead to early exit if the array is
    // sorted
    auto pivot = partition(first, last, nth);

    if (pivot == nth)
      return;

    if (nth < pivot)
      last = pivot;
    else
      first = pivot + 1;
  }
}

} // namespace ArborX::Details::KokkosExt

#endif
