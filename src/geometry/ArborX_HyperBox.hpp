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

#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_ReductionIdentity.hpp>

namespace ArborX::ExperimentalHyperGeometry
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

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  using type = BoxTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  using type = Coordinate;
};

template <int DIM, typename Coordinate>
struct [[deprecated]] Kokkos::reduction_identity<
    ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>>
{
  KOKKOS_FUNCTION static ArborX::ExperimentalHyperGeometry::Box<DIM, Coordinate>
  sum()
  {
    return {};
  }
};

#endif
