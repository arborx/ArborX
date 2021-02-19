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

#ifndef ARBORX_DETAILS_MORTON_CODE_UTILS_HPP
#define ARBORX_DETAILS_MORTON_CODE_UTILS_HPP

#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp> // min. max
#include <ArborX_Exception.hpp>

namespace ArborX
{

namespace Details
{

// Insert one 0 bit after each of the 16 low bits of x
KOKKOS_INLINE_FUNCTION
unsigned int expandBitsBy1(unsigned int x)
{
  x &= 0x0000ffff;
  x = (x ^ (x << 8)) & 0x00ff00ff;
  x = (x ^ (x << 4)) & 0x0f0f0f0f;
  x = (x ^ (x << 2)) & 0x33333333;
  x = (x ^ (x << 1)) & 0x55555555;
  return x;
}

// Insert two 0 bits after each of the 10 low bits of x
KOKKOS_INLINE_FUNCTION
unsigned int expandBitsBy2(unsigned int x)
{
  x &= 0x000003ff;
  x = (x ^ (x << 16)) & 0xff0000ff;
  x = (x ^ (x << 8)) & 0x0300f00f;
  x = (x ^ (x << 4)) & 0x030c30c3;
  x = (x ^ (x << 2)) & 0x09249249;
  return x;
}

// Calculates a 32-bit Morton code for a given 2D point located within the unit
// cube [0,1].
KOKKOS_INLINE_FUNCTION
unsigned int morton2D(float x, float y)
{
  // The interval [0,1] is subdivided into 65536 bins (in each direction).
  constexpr unsigned N = (1 << 16);

  using KokkosExt::max;
  using KokkosExt::min;

  x = min(max(x * N, 0.f), (float)N - 1);
  y = min(max(y * N, 0.f), (float)N - 1);

  return 2 * expandBitsBy1((unsigned int)x) + expandBitsBy1((unsigned int)y);
}

// Calculates a 30-bit Morton code for a
// given 3D point located within the unit cube [0,1].
KOKKOS_INLINE_FUNCTION
unsigned int morton3D(float x, float y, float z)
{
  // The interval [0,1] is subdivided into 1024 bins (in each direction).
  constexpr unsigned N = (1 << 10);

  using KokkosExt::max;
  using KokkosExt::min;

  x = min(max(x * N, 0.f), (float)N - 1);
  y = min(max(y * N, 0.f), (float)N - 1);
  z = min(max(z * N, 0.f), (float)N - 1);

  return 4 * expandBitsBy2((unsigned)x) + 2 * expandBitsBy2((unsigned)y) +
         expandBitsBy2((unsigned)z);
}

} // namespace Details

} // namespace ArborX

#endif
