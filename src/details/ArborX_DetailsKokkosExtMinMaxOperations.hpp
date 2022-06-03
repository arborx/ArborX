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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_MIN_MAX_OPERATIONS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_MIN_MAX_OPERATIONS_HPP

#include <Kokkos_Macros.hpp>

#include <initializer_list>

namespace KokkosExt
{

//! Compute the maximum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &max(T const &a, T const &b)
{
  return (a > b) ? a : b;
}

//! Compute the minimum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &min(T const &a, T const &b)
{
  return (a < b) ? a : b;
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T max(std::initializer_list<T> ilist)
{
  auto const *first = ilist.begin();
  auto const *const last = ilist.end();
  auto result = *first;
  if (first == last)
  {
    return result;
  }
  while (++first != last)
  {
    if (result < *first)
    {
      result = *first;
    }
  }
  return result;
}

template <class T>
KOKKOS_INLINE_FUNCTION constexpr T min(std::initializer_list<T> ilist)
{
  auto const *first = ilist.begin();
  auto const *const last = ilist.end();
  auto result = *first;
  if (first == last)
  {
    return result;
  }
  while (++first != last)
  {
    if (*first < result)
    {
      result = *first;
    }
  }
  return result;
}

} // namespace KokkosExt

#endif
