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
#ifndef ARBORX_HYPERSPHERE_HPP
#define ARBORX_HYPERSPHERE_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX::ExperimentalHyperGeometry
{

template <int DIM, class Coordinate = float>
struct Sphere
{
  KOKKOS_DEFAULTED_FUNCTION
  Sphere() = default;

  KOKKOS_FUNCTION
  constexpr Sphere(Point<DIM, Coordinate> const &centroid, Coordinate radius)
      : _centroid(centroid)
      , _radius(radius)
  {}

  KOKKOS_FUNCTION
  constexpr auto &centroid() { return _centroid; }

  KOKKOS_FUNCTION
  constexpr auto const &centroid() const { return _centroid; }

  KOKKOS_FUNCTION
  constexpr auto radius() const { return _radius; }

  Point<DIM, Coordinate> _centroid = {};
  Coordinate _radius = 0;
};

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Sphere<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Sphere<DIM, Coordinate>>
{
  using type = SphereTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Sphere<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
