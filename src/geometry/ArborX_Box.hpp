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

#ifndef ARBORX_BOX_HPP
#define ARBORX_BOX_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_ReductionIdentity.hpp>

#include <type_traits>

namespace ArborX
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
template <int DIM, class Coordinate = float>
struct Box
{
  KOKKOS_FUNCTION
  constexpr Box()
  {
    for (int d = 0; d < DIM; ++d)
    {
      _min_corner[d] =
          Details::KokkosExt::ArithmeticTraits::finite_max<Coordinate>::value;
      _max_corner[d] =
          Details::KokkosExt::ArithmeticTraits::finite_min<Coordinate>::value;
    }
  }

  KOKKOS_FUNCTION
  constexpr Box(Point<DIM, Coordinate> const &min_corner,
                Point<DIM, Coordinate> const &max_corner)
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

  Point<DIM, Coordinate> _min_corner;
  Point<DIM, Coordinate> _max_corner;
};

template <typename T, std::size_t N>
#if KOKKOS_VERSION >= 40400
KOKKOS_DEDUCTION_GUIDE
#else
KOKKOS_FUNCTION
#endif
Box(T const (&)[N], T const (&)[N]) -> Box<N, T>;

} // namespace ArborX

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<ArborX::Box<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<ArborX::Box<DIM, Coordinate>>
{
  using type = BoxTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<ArborX::Box<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
