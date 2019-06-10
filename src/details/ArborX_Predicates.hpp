/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_PREDICATE_HPP
#define ARBORX_PREDICATE_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsNode.hpp>

namespace ArborX
{
namespace Details
{
struct NearestPredicateTag
{
};
struct SpatialPredicateTag
{
};
} // namespace Details

template <typename Geometry>
struct Nearest
{
  using Tag = Details::NearestPredicateTag;

  KOKKOS_INLINE_FUNCTION
  Nearest() = default;

  KOKKOS_INLINE_FUNCTION
  Nearest(Geometry const &geometry, int k)
      : _geometry(geometry)
      , _k(k)
  {
  }

  Geometry _geometry;
  int _k = 0;
};

struct Sphere2
{
  KOKKOS_INLINE_FUNCTION
  Sphere2() = default;

  KOKKOS_INLINE_FUNCTION
  Sphere2(Point const &centroid, Details::DistanceReturnType radius)
      : _centroid(centroid)
      , _radius(radius)
  {
  }

  KOKKOS_INLINE_FUNCTION
  Sphere2(Point const &centroid, double radius)
      : _centroid(centroid)
      , _radius(Details::DistanceReturnType{radius * radius})
  {
  }

  KOKKOS_INLINE_FUNCTION
  Point &centroid() { return _centroid; }

  KOKKOS_INLINE_FUNCTION
  Point const &centroid() const { return _centroid; }

  KOKKOS_INLINE_FUNCTION
  Details::DistanceReturnType radius() const { return _radius; }

  Point _centroid;
  Details::DistanceReturnType _radius = Details::DistanceReturnType{0.};
};

namespace Details
{

KOKKOS_INLINE_FUNCTION
bool intersects(Sphere2 const &sphere, Box const &box)
{
  return Details::distance(sphere.centroid(), box) <= sphere.radius();
}

KOKKOS_INLINE_FUNCTION
Point returnCentroid(Sphere2 const &sphere) { return sphere.centroid(); }

KOKKOS_INLINE_FUNCTION
void expand(Box &box, Sphere2 const &sphere)
{
  for (int d = 0; d < 3; ++d)
  {
    box.minCorner()[d] = std::min<double>(
        box.minCorner()[d], sphere.centroid()[d] - sphere.radius().to_double());
    box.maxCorner()[d] = std::max<double>(
        box.maxCorner()[d], sphere.centroid()[d] + sphere.radius().to_double());
  }
}

} // namespace Details

template <typename Geometry>
struct Intersects
{
  using Tag = Details::SpatialPredicateTag;

  KOKKOS_INLINE_FUNCTION Intersects() = default;

  KOKKOS_INLINE_FUNCTION Intersects(Geometry const &geometry)
      : _geometry(geometry)
  {
  }

  template <typename Other>
  KOKKOS_INLINE_FUNCTION bool operator()(Other const &other) const
  {
    return Details::intersects(_geometry, other);
  }

  Geometry _geometry;
};

using Within = Intersects<Sphere2>;
using Overlap = Intersects<Box>;

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Nearest<Geometry> nearest(Geometry const &geometry,
                                                 int k = 1)
{
  return Nearest<Geometry>(geometry, k);
}

KOKKOS_INLINE_FUNCTION
Within within(Point const &p, double r)
{
  return Within(Sphere2{p, Details::DistanceReturnType{r * r}});
}

KOKKOS_INLINE_FUNCTION
Within within(Point const &p, Details::DistanceReturnType r)
{
  return Within(Sphere2{p, r});
}

KOKKOS_INLINE_FUNCTION
Overlap overlap(Box const &b) { return Overlap(b); }

} // namespace ArborX

#endif
