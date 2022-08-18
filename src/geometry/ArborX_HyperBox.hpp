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

#ifndef ARBORX_HYPERBOX_HPP
#define ARBORX_HYPERBOX_HPP

#if KOKKOS_VERSION >= 30700

#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::Experimental
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
template <int DIM, class FloatingPoint = float>
struct Box
{
  KOKKOS_FUNCTION
  constexpr Box()
  {
    for (int d = 0; d < DIM; ++d)
    {
      _min_corner[d] =
          KokkosExt::ArithmeticTraits::finite_max<FloatingPoint>::value;
      _max_corner[d] =
          KokkosExt::ArithmeticTraits::finite_min<FloatingPoint>::value;
    }
  }

  KOKKOS_FUNCTION
  constexpr Box(Point<DIM, FloatingPoint> const &min_corner,
                Point<DIM, FloatingPoint> const &max_corner)
      : _min_corner(min_corner)
      , _max_corner(max_corner)
  {}

  KOKKOS_FUNCTION
  constexpr auto &minCorner() { return _min_corner; }

  KOKKOS_FUNCTION
  constexpr auto const &minCorner() const { return _min_corner; }

  KOKKOS_FUNCTION
  constexpr auto &maxCorner() { return _max_corner; }

  KOKKOS_FUNCTION
  constexpr auto const &maxCorner() const { return _max_corner; }

  Point<DIM, FloatingPoint> _min_corner;
  Point<DIM, FloatingPoint> _max_corner;

  KOKKOS_FUNCTION auto &operator+=(Box const &other)
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

  KOKKOS_FUNCTION auto &operator+=(Point<DIM, FloatingPoint> const &point)
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
};

} // namespace ArborX::Experimental

template <int DIM, class FloatingPoint>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::Box<DIM, FloatingPoint>>
{
  static constexpr int value = DIM;
};
template <int DIM, class FloatingPoint>
struct ArborX::GeometryTraits::tag<
    ArborX::Experimental::Box<DIM, FloatingPoint>>
{
  using type = BoxTag;
};

#endif

#endif
