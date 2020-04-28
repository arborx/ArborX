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

#ifndef ARBORX_DETAILS_MORTON_CODE_UTILS_HPP
#define ARBORX_DETAILS_MORTON_CODE_UTILS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsKokkosExt.hpp> // min. max
#include <ArborX_Exception.hpp>

#include <filling_curves.hh>

namespace ArborX
{

namespace Details
{

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
KOKKOS_INLINE_FUNCTION
unsigned int expandBits(unsigned int v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
KOKKOS_INLINE_FUNCTION
unsigned int morton3D(double x, double y, double z)
{
  using KokkosExt::max;
  using KokkosExt::min;

  // The interval [0,1] is subdivided into 1024 bins (in each direction).
  // If we were to use more bits to encode the Morton code, we would need
  // to reflect these changes in expandBits() as well as in the clz()
  // function that returns the number of leading zero bits since it
  // currently assumes that the code can be represented by a 32 bit
  // integer.
  x = min(max(x * 1024.0, 0.0), 1023.0);
  y = min(max(y * 1024.0, 0.0), 1023.0);
  z = min(max(z * 1024.0, 0.0), 1023.0);
  unsigned int xx = expandBits((unsigned int)x);
  unsigned int yy = expandBits((unsigned int)y);
  unsigned int zz = expandBits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

unsigned int flecsi_hilbert_proj(Box const &b, Point const &p)
{
  using point_t = flecsi::space_vector_u<double, 3>;
  using range_t = std::array<point_t, 2>;
  auto a2f = [](Point const &x) -> point_t { return {x[0], x[1], x[2]}; };
  range_t range = {a2f(b.minCorner()), a2f(b.maxCorner())};
  return flecsi::hilbert_curve_u<3, uint32_t>{range, a2f(p)}.value();
}

} // namespace Details

} // namespace ArborX

#endif
