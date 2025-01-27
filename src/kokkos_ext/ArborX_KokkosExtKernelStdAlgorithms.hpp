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
#ifndef ARBORX_KOKKOS_EXT_KERNEL_STD_ALGORITHMS_HPP
#define ARBORX_KOKKOS_EXT_KERNEL_STD_ALGORITHMS_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Swap.hpp>

namespace ArborX::Details::KokkosExt
{

template <typename Iterator>
KOKKOS_FUNCTION void nth_element(Iterator first, Iterator nth, Iterator last)
{
  if (first == last || nth == last)
    return;

  // Lomuto partitioning
  auto partition = [](Iterator left, Iterator right, Iterator pivot) {
    using Kokkos::kokkos_swap;

    --right;

    Kokkos::kokkos_swap(*pivot, *right);
    auto it_i = left;
    auto it_j = left;
    while (it_j < right)
    {
      if (*it_j < *right)
        kokkos_swap(*it_j, *(it_i++));
      ++it_j;
    }
    kokkos_swap(*it_i, *right);
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

template <typename Iterator, typename T>
KOKKOS_FUNCTION Iterator upper_bound(Iterator first, Iterator last,
                                     T const &value)
{
  int count = last - first;
  while (count > 0)
  {
    int step = count / 2;
    if (!(value < *(first + step)))
    {
      first += step + 1;
      count -= step + 1;
    }
    else
    {
      count = step;
    }
  }
  return first;
}

} // namespace ArborX::Details::KokkosExt

#endif
