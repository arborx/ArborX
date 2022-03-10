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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_MATH_FUNCTIONS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_MATH_FUNCTIONS_HPP

#include <Kokkos_MathematicalFunctions.hpp>

namespace KokkosExt
{

#if KOKKOS_VERSION >= 30699
using Kokkos::isfinite;

KOKKOS_INLINE_FUNCTION float hypot(float x, float y, float z)
{
  return Kokkos::sqrt(x * x + y * y + z * z);
}
#else
using Kokkos::Experimental::isfinite;

KOKKOS_INLINE_FUNCTION float hypot(float x, float y, float z)
{
  return Kokkos::Experimental::sqrt(x * x + y * y + z * z);
}
#endif

} // namespace KokkosExt

#endif
