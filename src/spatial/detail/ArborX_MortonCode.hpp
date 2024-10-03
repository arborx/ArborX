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

#ifndef ARBORX_MORTON_CODE_UTILS_HPP
#define ARBORX_MORTON_CODE_UTILS_HPP

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Abort.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>

namespace ArborX
{

namespace Details
{

// Magic numbers generated by the script in
// https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints/18528775#18528775

template <int N>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy(unsigned int)
{
  static_assert(0 <= N && N < 10,
                "expandBitsBy can only be used with values 0-9");
  Kokkos::abort("ArborX: implementation bug");
  return 0;
}

template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<0>(unsigned int x)
{
  return x;
}

// Insert one 0 bit after each of the 16 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<1>(unsigned int x)
{
  x &= 0x0000ffffu;
  x = (x ^ (x << 8)) & 0x00ff00ffu;
  x = (x ^ (x << 4)) & 0x0f0f0f0fu;
  x = (x ^ (x << 2)) & 0x33333333u;
  x = (x ^ (x << 1)) & 0x55555555u;
  return x;
}

// Insert two 0 bits after each of the 10 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<2>(unsigned int x)
{
  x &= 0x000003ffu;
  x = (x ^ (x << 16)) & 0xff0000ffu;
  x = (x ^ (x << 8)) & 0x0300f00fu;
  x = (x ^ (x << 4)) & 0x030c30c3u;
  x = (x ^ (x << 2)) & 0x09249249u;
  return x;
}

// Insert three 0 bits after each of the 8 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<3>(unsigned int x)
{
  x &= 0xffu;
  x = (x | x << 16) & 0xc0003fu;
  x = (x | x << 8) & 0xc03807u;
  x = (x | x << 4) & 0x8430843u;
  x = (x | x << 2) & 0x9090909u;
  x = (x | x << 1) & 0x11111111u;
  return x;
}

// Insert four 0 bits after each of the 6 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<4>(unsigned int x)
{
  x &= 0x3fu;
  x = (x | x << 16) & 0x30000fu;
  x = (x | x << 8) & 0x300c03u;
  x = (x | x << 4) & 0x2108421u;
  return x;
}

// Insert five 0 bits after each of the 5 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<5>(unsigned int x)
{
  x &= 0x1fu;
  x = (x | x << 16) & 0x10000fu;
  x = (x | x << 8) & 0x100c03u;
  x = (x | x << 4) & 0x1008421u;
  x = (x | x << 2) & 0x1021021u;
  x = (x | x << 1) & 0x1041041u;
  return x;
}

// Insert six 0 bits after each of the 4 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<6>(unsigned int x)
{
  x &= 0xfu;
  x = (x | x << 16) & 0x80007u;
  x = (x | x << 8) & 0x80403u;
  x = (x | x << 4) & 0x84021u;
  x = (x | x << 2) & 0x204081u;
  return x;
}

// Insert seven 0 bits after each of the 4 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<7>(unsigned int x)
{
  x &= 0xfu;
  x = (x | x << 16) & 0x80007u;
  x = (x | x << 8) & 0x80403u;
  x = (x | x << 4) & 0x804021u;
  x = (x | x << 2) & 0x810081u;
  x = (x | x << 1) & 0x1010101u;
  return x;
}

// Insert eight 0 bits after each of the 3 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<8>(unsigned int x)
{
  x &= 0x7u;
  x = (x | x << 16) & 0x40003u;
  x = (x | x << 8) & 0x40201u;
  return x;
}

// Insert nine 0 bits after each of the 3 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned int expandBitsBy<9>(unsigned int x)
{
  x &= 0x7u;
  x = (x | x << 16) & 0x40003u;
  x = (x | x << 8) & 0x40201u;
  x = (x | x << 2) & 0x100201u;
  x = (x | x << 1) & 0x100401u;
  return x;
}

template <int N>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy(unsigned long long)
{
  static_assert(0 <= N && N < 10,
                "expandBitsBy can only be used with values 0-9");
  Kokkos::abort("ArborX: implementation bug");
  return 0;
}

template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<0>(unsigned long long x)
{
  return x;
}

// Insert one 0 bit after each of the 31 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<1>(unsigned long long x)
{
  x &= 0x7fffffffllu;
  x = (x | x << 16) & 0x7fff0000ffffllu;
  x = (x | x << 8) & 0x7f00ff00ff00ffllu;
  x = (x | x << 4) & 0x70f0f0f0f0f0f0fllu;
  x = (x | x << 2) & 0x1333333333333333llu;
  x = (x | x << 1) & 0x1555555555555555llu;
  return x;
}

// Insert two 0 bits after each of the 21 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<2>(unsigned long long x)
{
  x &= 0x1fffffllu;
  x = (x | x << 32) & 0x1f00000000ffffllu;
  x = (x | x << 16) & 0x1f0000ff0000ffllu;
  x = (x | x << 8) & 0x100f00f00f00f00fllu;
  x = (x | x << 4) & 0x10c30c30c30c30c3llu;
  x = (x | x << 2) & 0x1249249249249249llu;
  return x;
}

// Insert three 0 bits after each of the 15 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<3>(unsigned long long x)
{
  x &= 0x7fffllu;
  x = (x | x << 32) & 0x7800000007ffllu;
  x = (x | x << 16) & 0x780007c0003fllu;
  x = (x | x << 8) & 0x40380700c03807llu;
  x = (x | x << 4) & 0x43084308430843llu;
  x = (x | x << 2) & 0x109090909090909llu;
  x = (x | x << 1) & 0x111111111111111llu;
  return x;
}

// Insert four 0 bits after each of the 12 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<4>(unsigned long long x)
{
  x &= 0xfffllu;
  x = (x | x << 32) & 0xf00000000ffllu;
  x = (x | x << 16) & 0xf0000f0000fllu;
  x = (x | x << 8) & 0xc0300c0300c03llu;
  x = (x | x << 4) & 0x84210842108421llu;
  return x;
}

// Insert five 0 bits after each of the 10 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<5>(unsigned long long x)
{
  x &= 0x3ffllu;
  x = (x | x << 32) & 0x3800000007fllu;
  x = (x | x << 16) & 0x3800070000fllu;
  x = (x | x << 8) & 0x3008060100c03llu;
  x = (x | x << 4) & 0x21008421008421llu;
  x = (x | x << 2) & 0x21021021021021llu;
  x = (x | x << 1) & 0x41041041041041llu;
  return x;
}

// Insert six 0 bits after each of the 9 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<6>(unsigned long long x)
{
  x &= 0x1ffllu;
  x = (x | x << 32) & 0x1c00000003fllu;
  x = (x | x << 16) & 0x10000c000380007llu;
  x = (x | x << 8) & 0x100804030080403llu;
  x = (x | x << 4) & 0x100840210084021llu;
  x = (x | x << 2) & 0x102040810204081llu;
  return x;
}

// Insert seven 0 bits after each of the 7 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<7>(unsigned long long x)
{
  x &= 0x7fllu;
  x = (x | x << 32) & 0x600000001fllu;
  x = (x | x << 16) & 0x6000180007llu;
  x = (x | x << 8) & 0x402010080403llu;
  x = (x | x << 4) & 0x402100804021llu;
  x = (x | x << 2) & 0x1008100810081llu;
  x = (x | x << 1) & 0x1010101010101llu;
  return x;
}

// Insert eight 0 bits after each of the 7 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<8>(unsigned long long x)
{
  x &= 0x7fllu;
  x = (x | x << 32) & 0x700000000fllu;
  x = (x | x << 16) & 0x400030000c0003llu;
  x = (x | x << 8) & 0x40201008040201llu;
  return x;
}

// Insert nine 0 bits after each of the 6 low bits of x
template <>
KOKKOS_INLINE_FUNCTION unsigned long long expandBitsBy<9>(unsigned long long x)
{
  x &= 0x3fllu;
  x = (x | x << 32) & 0x300000000fllu;
  x = (x | x << 16) & 0x30000c0003llu;
  x = (x | x << 8) & 0x201008040201llu;
  x = (x | x << 4) & 0x2010008040201llu;
  x = (x | x << 2) & 0x2010020100201llu;
  x = (x | x << 1) & 0x4010040100401llu;
  return x;
}

template <typename Point,
          typename Enable = std::enable_if_t<GeometryTraits::is_point_v<Point>>>
KOKKOS_INLINE_FUNCTION unsigned int morton32(Point const &p)
{
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  constexpr unsigned N = 1u << (DIM == 1 ? 31 : 32 / DIM);

  using Kokkos::max;
  using Kokkos::min;

  unsigned int r = 0;
  for (int d = 0; d < DIM; ++d)
  {
    auto x = min(max((float)p[d] * N, 0.f), (float)N - 1);
    r += (expandBitsBy<DIM - 1>((unsigned int)x) << (DIM - d - 1));
  }

  return r;
}

template <typename Point,
          std::enable_if_t<GeometryTraits::is_point_v<Point> &&
                           GeometryTraits::dimension_v<Point> != 2> * = nullptr>
KOKKOS_INLINE_FUNCTION unsigned long long morton64(Point const &p)
{
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  constexpr unsigned long long N = (1llu << (63 / DIM));

  using Kokkos::max;
  using Kokkos::min;

  unsigned long long r = 0;
  for (int d = 0; d < DIM; ++d)
  {
    auto x = min(max((float)p[d] * N, 0.f), (float)N - 1);
    r += (expandBitsBy<DIM - 1>((unsigned long long)x) << (DIM - d - 1));
  }

  return r;
}

// Calculate a 62-bit Morton code for a 2D point located within [0, 1]^2.
// Special case because it needs double.
template <typename Point,
          std::enable_if_t<GeometryTraits::is_point_v<Point> &&
                           GeometryTraits::dimension_v<Point> == 2> * = nullptr>
KOKKOS_INLINE_FUNCTION unsigned long long morton64(Point const &p)
{
  // The interval [0,1] is subdivided into 2,147,483,648 bins (in each
  // direction).
  constexpr unsigned N = (1u << 31);

  using Kokkos::max;
  using Kokkos::min;

  // Have to use double as float is not sufficient to represent large
  // integers, which would result in some missing bins.
  auto xd = min(max((double)p[0] * N, 0.), (double)N - 1);
  auto yd = min(max((double)p[1] * N, 0.), (double)N - 1);

  return 2 * expandBitsBy<1>((unsigned long long)xd) +
         expandBitsBy<1>((unsigned long long)yd);
}

} // namespace Details

} // namespace ArborX

#endif
