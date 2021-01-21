/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

} // namespace KokkosExt

#endif
