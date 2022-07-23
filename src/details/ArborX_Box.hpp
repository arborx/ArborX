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

#ifndef ARBORX_BOX_HPP
#define ARBORX_BOX_HPP

#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
template <int DIM = 3>
struct BoxD
{
  static int const dim = DIM;

  KOKKOS_FUNCTION
  constexpr BoxD()
  {
    for (int d = 0; d < DIM; ++d)
    {
      _min_corner[d] = KokkosExt::ArithmeticTraits::finite_max<float>::value;
      _max_corner[d] = KokkosExt::ArithmeticTraits::finite_min<float>::value;
    }
  }

  KOKKOS_INLINE_FUNCTION
  constexpr BoxD(PointD<DIM> const &min_corner, PointD<DIM> const &max_corner)
      : _min_corner(min_corner)
      , _max_corner(max_corner)
  {}

  KOKKOS_INLINE_FUNCTION
  constexpr PointD<DIM> &minCorner() { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr PointD<DIM> const &minCorner() const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  PointD<DIM> volatile &minCorner() volatile { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  PointD<DIM> const volatile &minCorner() const volatile { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr PointD<DIM> &maxCorner() { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr PointD<DIM> const &maxCorner() const { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  PointD<DIM> volatile &maxCorner() volatile { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  PointD<DIM> const volatile &maxCorner() const volatile { return _max_corner; }

  PointD<DIM> _min_corner;
  PointD<DIM> _max_corner;

  KOKKOS_FUNCTION BoxD &operator+=(BoxD const &other)
  {
    using KokkosExt::max;
    using KokkosExt::min;

    for (int d = 0; d < DIM; ++d)
    {
      minCorner()[d] = min(minCorner()[d], other.minCorner()[d]);
      maxCorner()[d] = max(maxCorner()[d], other.maxCorner()[d]);
    }
    return *this;
  }

  KOKKOS_FUNCTION void operator+=(BoxD const volatile &other) volatile
  {
    using KokkosExt::max;
    using KokkosExt::min;

    for (int d = 0; d < DIM; ++d)
    {
      minCorner()[d] = min(minCorner()[d], other.minCorner()[d]);
      maxCorner()[d] = max(maxCorner()[d], other.maxCorner()[d]);
    }
  }

  KOKKOS_FUNCTION BoxD &operator+=(PointD<DIM> const &point)
  {
    using KokkosExt::max;
    using KokkosExt::min;

    for (int d = 0; d < DIM; ++d)
    {
      minCorner()[d] = min(minCorner()[d], point[d]);
      maxCorner()[d] = max(maxCorner()[d], point[d]);
    }
    return *this;
  }

// FIXME Temporary workaround until we clarify requirements on the Kokkos side.
#if defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_SYCL)
private:
  friend KOKKOS_FUNCTION BoxD operator+(BoxD box, BoxD const &other)
  {
    return box += other;
  }
#endif
};

using Box = BoxD<3>;

} // namespace ArborX

#endif
