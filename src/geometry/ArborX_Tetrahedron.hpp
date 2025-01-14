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

#ifndef ARBORX_TETRAHEDRON_HPP
#define ARBORX_TETRAHEDRON_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Array.hpp>

namespace ArborX::ExperimentalHyperGeometry
{
// Need to add a protection that the points are not in the same plane
template <class Coordinate = float>
struct Tetrahedron
{
  Point<3, Coordinate> a;
  Point<3, Coordinate> b;
  Point<3, Coordinate> c;
  Point<3, Coordinate> d;
};

template <class Coordinate>
Tetrahedron(Point<3, Coordinate>, Point<3, Coordinate>, Point<3, Coordinate>,
            Point<3, Coordinate>) -> Tetrahedron<Coordinate>;

} // namespace ArborX::ExperimentalHyperGeometry

template <class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>>
{
  static constexpr int value = 3;
};
template <class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>>
{
  using type = TetrahedronTag;
};
template <class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>>
{
  using type = Coordinate;
};

#endif
