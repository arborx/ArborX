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

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp> // isfinite

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

namespace ArborX::Details::Dispatch
{
using GeometryTraits::BoxTag;
using GeometryTraits::PointTag;
using GeometryTraits::SphereTag;

// equals sphere-sphere
template <typename Sphere>
struct equals<SphereTag, Sphere>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &l, Sphere const &r)
  {
    return Details::equals(l.centroid(), r.centroid()) &&
           l.radius() == r.radius();
  }
};

// isValid sphere
template <typename Sphere>
struct isValid<SphereTag, Sphere>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &s)
  {
    return Details::isValid(s.centroid()) && Kokkos::isfinite(s.radius()) &&
           (s.radius() >= 0.);
  }
};

// distance point-sphere
template <typename Point, typename Sphere>
struct distance<PointTag, SphereTag, Point, Sphere>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Sphere const &sphere)
  {
    using KokkosExt::max;
    return max(Details::distance(point, sphere.centroid()) - sphere.radius(),
               0.f);
  }
};

// distance sphere-box
template <typename Sphere, typename Box>
struct distance<SphereTag, BoxTag, Sphere, Box>
{
  KOKKOS_FUNCTION static auto apply(Sphere const &sphere, Box const &box)
  {
    using KokkosExt::max;

    auto distance_center_box = Details::distance(sphere.centroid(), box);
    return max(distance_center_box - sphere.radius(), 0.f);
  }
};

// expand a box to include a sphere
template <typename Box, typename Sphere>
struct expand<BoxTag, SphereTag, Box, Sphere>
{
  KOKKOS_FUNCTION static void apply(Box &box, Sphere const &sphere)
  {
    using KokkosExt::max;
    using KokkosExt::min;

    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      box.minCorner()[d] =
          min(box.minCorner()[d], sphere.centroid()[d] - sphere.radius());
      box.maxCorner()[d] =
          max(box.maxCorner()[d], sphere.centroid()[d] + sphere.radius());
    }
  }
};

// check if a sphere intersects with an axis-aligned bounding box
template <typename Sphere, typename Box>
struct intersects<SphereTag, BoxTag, Sphere, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &sphere,
                                              Box const &box)
  {
    return Details::distance(sphere.centroid(), box) <= sphere.radius();
  }
};

// check if a sphere intersects with a point
template <typename Sphere, typename Point>
struct intersects<SphereTag, PointTag, Sphere, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &sphere,
                                              Point const &point)
  {
    return Details::distance(sphere.centroid(), point) <= sphere.radius();
  }
};

template <typename Point, typename Sphere>
struct intersects<PointTag, SphereTag, Point, Sphere>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              Sphere const &sphere)
  {
    return Details::intersects(sphere, point);
  }
};

template <typename Sphere>
struct centroid<SphereTag, Sphere>
{
  KOKKOS_FUNCTION static constexpr auto apply(Sphere const &sphere)
  {
    return sphere.centroid();
  }
};

} // namespace ArborX::Details::Dispatch

#endif
