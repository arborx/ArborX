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
#ifndef ARBORX_ELLIPSOID_HPP
#define ARBORX_ELLIPSOID_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

#include <initializer_list>
#include <type_traits>

namespace ArborX::Experimental
{

template <int DIM, class Coordinate = float>
struct Ellipsoid
{
  KOKKOS_DEFAULTED_FUNCTION
  Ellipsoid() = default;

  KOKKOS_FUNCTION
  constexpr Ellipsoid(Point<DIM, Coordinate> const &centroid,
                      Coordinate const rmt[DIM][DIM])
      : _centroid(centroid)
  {
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        _rmt[i][j] = rmt[i][j];
  }

  KOKKOS_FUNCTION
  constexpr Ellipsoid(
      Point<DIM, Coordinate> const &centroid,
      std::initializer_list<std::initializer_list<Coordinate>> const rmt)
      : _centroid(centroid)
  {
    KOKKOS_ASSERT(rmt.size() == DIM);
    int i = 0;
    for (auto const &row : rmt)
    {
      KOKKOS_ASSERT(row.size() == DIM);
      int j = 0;
      for (auto const &value : row)
        _rmt[i][j++] = value;
      ++i;
    }
  }

  KOKKOS_FUNCTION
  constexpr auto &centroid() { return _centroid; }

  KOKKOS_FUNCTION
  constexpr auto const &centroid() const { return _centroid; }

  KOKKOS_FUNCTION
  constexpr auto const &rmt() const { return _rmt; }

  Point<DIM, Coordinate> _centroid = {};
  Coordinate _rmt[DIM][DIM] = {};
};

template <typename T, std::size_t N>
#if KOKKOS_VERSION >= 40400
KOKKOS_DEDUCTION_GUIDE
#else
KOKKOS_FUNCTION
#endif
Ellipsoid(T const (&)[N], T const (&)[N][N]) -> Ellipsoid<N, T>;

} // namespace ArborX::Experimental

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::Ellipsoid<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::Experimental::Ellipsoid<DIM, Coordinate>>
{
  using type = EllipsoidTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::Ellipsoid<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
