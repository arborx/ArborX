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
#ifndef ARBORX_DETAILS_ALGORITHMS_HPP
#define ARBORX_DETAILS_ALGORITHMS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp> // min, max
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Macros.hpp>
#if KOKKOS_VERSION >= 40200
#include <Kokkos_Assert.hpp> // KOKKOS_ASSERT
#else
#include <impl/Kokkos_Error.hpp> // KOKKOS_ASSERT
#endif
#include <Kokkos_MathematicalFunctions.hpp> // isfinite

namespace ArborX
{
namespace Details
{

namespace Dispatch
{
using GeometryTraits::BoxTag;
using GeometryTraits::KDOPTag;
using GeometryTraits::PointTag;
using GeometryTraits::SphereTag;
using GeometryTraits::TriangleTag;

template <typename Tag, typename Geometry>
struct equals;

template <typename Tag, typename Geometry>
struct isValid;

template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct distance;

template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct expand;

template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct intersects;

template <typename Tag, typename Geometry>
struct centroid;

} // namespace Dispatch

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool equals(Geometry const &l,
                                             Geometry const &r)
{
  return Dispatch::equals<typename GeometryTraits::tag<Geometry>::type,
                          Geometry>::apply(l, r);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool isValid(Geometry const &geometry)
{
  return Dispatch::isValid<typename GeometryTraits::tag<Geometry>::type,
                           Geometry>::apply(geometry);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION auto distance(Geometry1 const &geometry1,
                                     Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::distance<typename GeometryTraits::tag<Geometry1>::type,
                            typename GeometryTraits::tag<Geometry2>::type,
                            Geometry1, Geometry2>::apply(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION void expand(Geometry1 &geometry1,
                                   Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  Dispatch::expand<typename GeometryTraits::tag<Geometry1>::type,
                   typename GeometryTraits::tag<Geometry2>::type, Geometry1,
                   Geometry2>::apply(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION constexpr bool intersects(Geometry1 const &geometry1,
                                                 Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::intersects<typename GeometryTraits::tag<Geometry1>::type,
                              typename GeometryTraits::tag<Geometry2>::type,
                              Geometry1, Geometry2>::apply(geometry1,
                                                           geometry2);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION decltype(auto) returnCentroid(Geometry const &geometry)
{
  return Dispatch::centroid<typename GeometryTraits::tag<Geometry>::type,
                            Geometry>::apply(geometry);
}

namespace Dispatch
{

// equals point-point
template <typename Point>
struct equals<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &l, Point const &r)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (l[d] != r[d])
        return false;
    return true;
  }
};

// equals box-box
template <typename Box>
struct equals<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &l, Box const &r)
  {
    return Details::equals(l.minCorner(), r.minCorner()) &&
           Details::equals(l.maxCorner(), r.maxCorner());
  }
};

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

// isValid point
template <typename Point>
struct isValid<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &p)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (!Kokkos::isfinite(p[d]))
        return false;
    return true;
  }
};

// isValid box
template <typename Box>
struct isValid<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
      if (r_d <= 0 || !Kokkos::isfinite(r_d))
        return false;
    }
    return true;
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

// distance point-point
template <typename Point1, typename Point2>
struct distance<PointTag, PointTag, Point1, Point2>
{
  KOKKOS_FUNCTION static auto apply(Point1 const &a, Point2 const &b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point1>;
    // Points may have different coordinate types. Try using implicit
    // conversion to get the best one.
    using Coordinate = decltype(b[0] - a[0]);
    Coordinate distance_squared = 0;
    for (int d = 0; d < DIM; ++d)
    {
      auto tmp = b[d] - a[d];
      distance_squared += tmp * tmp;
    }
    return Kokkos::sqrt(distance_squared);
  }
};

// distance point-box
template <typename Point, typename Box>
struct distance<PointTag, BoxTag, Point, Box>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    Point projected_point;
    for (int d = 0; d < DIM; ++d)
    {
      if (point[d] < box.minCorner()[d])
        projected_point[d] = box.minCorner()[d];
      else if (point[d] > box.maxCorner()[d])
        projected_point[d] = box.maxCorner()[d];
      else
        projected_point[d] = point[d];
    }
    return Details::distance(point, projected_point);
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

// distance box-box
template <typename Box1, typename Box2>
struct distance<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static auto apply(Box1 const &box_a, Box2 const &box_b)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box1>;
    // Boxes may have different coordinate types. Try using implicit
    // conversion to get the best one.
    using Coordinate = decltype(box_b.minCorner()[0] - box_a.minCorner()[0]);
    Coordinate distance_squared = 0;
    for (int d = 0; d < DIM; ++d)
    {
      auto const a_min = box_a.minCorner()[d];
      auto const a_max = box_a.maxCorner()[d];
      auto const b_min = box_b.minCorner()[d];
      auto const b_max = box_b.maxCorner()[d];
      if (a_min > b_max)
      {
        auto const delta = a_min - b_max;
        distance_squared += delta * delta;
      }
      else if (b_min > a_max)
      {
        auto const delta = b_min - a_max;
        distance_squared += delta * delta;
      }
      else
      {
        // The boxes overlap on this axis: distance along this axis is zero.
      }
    }
    return Kokkos::sqrt(distance_squared);
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

// expand a box to include a point
template <typename Box, typename Point>
struct expand<BoxTag, PointTag, Box, Point>
{
  KOKKOS_FUNCTION static void apply(Box &box, Point const &point)
  {
    box += point;
  }
};

// expand a box to include a point
template <typename Box, typename KDOP>
struct expand<BoxTag, KDOPTag, Box, KDOP>
{
  KOKKOS_FUNCTION static void apply(Box &box, KDOP const &kdop)
  {
    // FIXME This is a workaround so that we can use existing conversion
    // machinery for KDOP. In the long term, this should be replaced by a
    // general algorithm.
    Details::expand(box, (ArborX::Box)kdop);
  }
};

// expand a box to include a box
template <typename Box1, typename Box2>
struct expand<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static void apply(Box1 &box, Box2 const &other)
  {
    box += other;
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

// expand a box to include a triangle
template <typename Box, typename Triangle>
struct expand<BoxTag, TriangleTag, Box, Triangle>
{
  KOKKOS_FUNCTION static void apply(Box &box, Triangle const &triangle)
  {
    Details::expand(box, triangle.a);
    Details::expand(box, triangle.b);
    Details::expand(box, triangle.c);
  }
};

// check if two axis-aligned bounding boxes intersect
template <typename Box1, typename Box2>
struct intersects<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box1 const &box,
                                              Box2 const &other)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box1>;
    for (int d = 0; d < DIM; ++d)
      if (box.minCorner()[d] > other.maxCorner()[d] ||
          box.maxCorner()[d] < other.minCorner()[d])
        return false;
    return true;
  }
};

// check it a box intersects with a point
template <typename Point, typename Box>
struct intersects<PointTag, BoxTag, Point, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              Box const &other)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (point[d] > other.maxCorner()[d] || point[d] < other.minCorner()[d])
        return false;
    return true;
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

template <typename Point, typename Triangle>
struct intersects<PointTag, TriangleTag, Point, Triangle>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              Triangle const &triangle)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    static_assert(DIM == 2);

    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    using Float = typename GeometryTraits::coordinate_type<Point>::type;

    // Find coefficients alpha and beta such that
    // x = a + alpha * (b - a) + beta * (c - a)
    //   = (1 - alpha - beta) * a + alpha * b + beta * c
    // recognizing the linear system
    // ((b - a) (c - a)) (alpha beta)^T = (x - a)
    Float u[] = {b[0] - a[0], b[1] - a[1]};
    Float v[] = {c[0] - a[0], c[1] - a[1]};
    Float const det = v[1] * u[0] - v[0] * u[1];
    KOKKOS_ASSERT(det != 0);
    Float const inv_det = 1 / det;

    Float alpha[] = {v[1] * inv_det, -v[0] * inv_det};
    Float beta[] = {-u[1] * inv_det, u[0] * inv_det};

    Float alpha_coeff =
        alpha[0] * (point[0] - a[0]) + alpha[1] * (point[1] - a[1]);
    Float beta_coeff =
        beta[0] * (point[0] - a[0]) + beta[1] * (point[1] - a[1]);

    Float coeffs[] = {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
    return (coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0);
  }
};

template <typename Point>
struct centroid<PointTag, Point>
{
  KOKKOS_FUNCTION static constexpr auto apply(Point const &point)
  {
    return point;
  }
};

template <typename Box>
struct centroid<BoxTag, Box>
{
  KOKKOS_FUNCTION static constexpr auto apply(Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    auto c = box.minCorner();
    for (int d = 0; d < DIM; ++d)
      c[d] = (c[d] + box.maxCorner()[d]) / 2;
    return c;
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

template <typename Triangle>
struct centroid<TriangleTag, Triangle>
{
  KOKKOS_FUNCTION static constexpr auto apply(Triangle const &triangle)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Triangle>;
    auto c = triangle.a;
    for (int d = 0; d < DIM; ++d)
      c[d] = (c[d] + triangle.b[d] + triangle.c[d]) / 3;
    return c;
  }
};

} // namespace Dispatch

// transformation that maps the unit cube into a new axis-aligned box
// NOTE safe to perform in-place
template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point<Point>{} &&
                           GeometryTraits::is_box<Box>{}> * = nullptr>
KOKKOS_FUNCTION void translateAndScale(Point const &in, Point &out,
                                       Box const &ref)
{
  static_assert(GeometryTraits::dimension_v<Point> ==
                GeometryTraits::dimension_v<Box>);
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  for (int d = 0; d < DIM; ++d)
  {
    auto const a = ref.minCorner()[d];
    auto const b = ref.maxCorner()[d];
    out[d] = (a != b ? (in[d] - a) / (b - a) : 0);
  }
}

} // namespace Details
} // namespace ArborX

#endif
