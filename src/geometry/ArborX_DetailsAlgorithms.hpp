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
#include <ArborX_DetailsVector.hpp>
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Assert.hpp> // KOKKOS_ASSERT
#include <Kokkos_Macros.hpp>
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
using GeometryTraits::RayTag;
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
  return Dispatch::equals<typename GeometryTraits::tag_t<Geometry>,
                          Geometry>::apply(l, r);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION constexpr bool isValid(Geometry const &geometry)
{
  return Dispatch::isValid<typename GeometryTraits::tag_t<Geometry>,
                           Geometry>::apply(geometry);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION auto distance(Geometry1 const &geometry1,
                                     Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::distance<typename GeometryTraits::tag_t<Geometry1>,
                            typename GeometryTraits::tag_t<Geometry2>,
                            Geometry1, Geometry2>::apply(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION void expand(Geometry1 &geometry1,
                                   Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  Dispatch::expand<typename GeometryTraits::tag_t<Geometry1>,
                   typename GeometryTraits::tag_t<Geometry2>, Geometry1,
                   Geometry2>::apply(geometry1, geometry2);
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION constexpr bool intersects(Geometry1 const &geometry1,
                                                 Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::intersects<typename GeometryTraits::tag_t<Geometry1>,
                              typename GeometryTraits::tag_t<Geometry2>,
                              Geometry1, Geometry2>::apply(geometry1,
                                                           geometry2);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION decltype(auto) returnCentroid(Geometry const &geometry)
{
  return Dispatch::centroid<typename GeometryTraits::tag_t<Geometry>,
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

// distance point-triangle
template <typename Point, typename Triangle>
struct distance<PointTag, TriangleTag, Point, Triangle>
{
  static constexpr int DIM = GeometryTraits::dimension_v<Point>;
  using Coordinate = GeometryTraits::coordinate_type_t<Triangle>;

  static_assert(DIM == 2 || DIM == 3);

  KOKKOS_FUNCTION static auto combine(Point const &a, Point const &b,
                                      Point const &c, Coordinate u,
                                      Coordinate v, Coordinate w)
  {
    Point r;
    for (int d = 0; d < DIM; ++d)
      r[d] = u * a[d] + v * b[d] + w * c[d];
    return r;
  }
  KOKKOS_FUNCTION static auto closest_point(Point const &p, Point const &a,
                                            Point const &b, Point const &c)
  {
    /* Zones
           \ 2/
            \/
        5   /\b  6
           /  \
          /    \
      \  /   0  \  /
       \/a______c\/
      1 |    4   | 3
        |        |
    */

    auto const ab = b - a;
    auto const ac = c - a;
    auto const ap = p - a;

    auto const d1 = ab.dot(ap);
    auto const d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) // zone 1
      return a;

    auto const bp = p - b;
    auto const d3 = ab.dot(bp);
    auto const d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) // zone 2
      return b;

    auto const cp = p - c;
    auto const d5 = ab.dot(cp);
    auto const d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6) // zone 3
      return c;

    auto const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) // zone 5
    {
      auto const v = d1 / (d1 - d3);
      return combine(a, b, c, 1 - v, v, 0);
    }

    auto const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) // zone 4
    {
      auto const v = d2 / (d2 - d6);
      return combine(a, b, c, 1 - v, 0, v);
    }

    auto const va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) // zone 6
    {
      auto const v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return combine(a, b, c, 0, 1 - v, v);
    }

    // zone 0
    auto const denom = 1 / (va + vb + vc);
    auto const v = vb * denom;
    auto const w = vc * denom;
    return combine(a, b, c, 1 - v - w, v, w);
  }

  KOKKOS_FUNCTION static auto apply(Point const &p, Triangle const &triangle)
  {
    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    return Details::distance(p, closest_point(p, a, b, c));
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
    using Details::KokkosExt::max;
    using Details::KokkosExt::min;

    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      box.minCorner()[d] = min(box.minCorner()[d], point[d]);
      box.maxCorner()[d] = max(box.maxCorner()[d], point[d]);
    }
  }
};

// expand a box to include a box
template <typename Box1, typename Box2>
struct expand<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static void apply(Box1 &box, Box2 const &other)
  {
    using Details::KokkosExt::max;
    using Details::KokkosExt::min;

    constexpr int DIM = GeometryTraits::dimension_v<Box1>;
    for (int d = 0; d < DIM; ++d)
    {
      box.minCorner()[d] = min(box.minCorner()[d], other.minCorner()[d]);
      box.maxCorner()[d] = max(box.maxCorner()[d], other.maxCorner()[d]);
    }
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

    using Float = typename GeometryTraits::coordinate_type_t<Point>;

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

template <typename Box, typename Triangle>
struct intersects<BoxTag, TriangleTag, Box, Triangle>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &box,
                                              Triangle const &triangle)
  {
    // Based on the Separating Axis Theorem
    // https://doi.org/10.1145/1198555.1198747
    // we have to project the box and the triangle onto 13 axes and check for
    // overlap. These axes are:
    // - the 3 normals of the box
    // - the normal of the triangle
    // - the 9 crossproduct between the 3 edge (directions) of the box and the 3
    // edges of the triangle

    // Testing the normals of the box is the same as testing the overlap of
    // bounding boxes.
    Box bounding_box;
    ArborX::Details::expand(bounding_box, triangle);
    if (!ArborX::Details::intersects(bounding_box, box))
      return false;

    // shift box and triangle so that the box's center is at the origin to
    // simplify the following checks.
    constexpr int DIM = GeometryTraits::dimension_v<Triangle>;
    static_assert(DIM == 3,
                  "Box-Triangle intersection only implemented in 3d!");
    auto min_corner = box.minCorner();
    auto max_corner = box.maxCorner();
    auto a = triangle.a;
    auto b = triangle.b;
    auto c = triangle.c;
    for (int i = 0; i < DIM; ++i)
    {
      auto const shift = -(max_corner[i] + min_corner[i]) / 2;
      a[i] += shift;
      b[i] += shift;
      c[i] += shift;
    }

    using Point = decltype(a);
    auto const vector_ab = b - a;
    auto const vector_ac = c - a;
    Point extents{(max_corner[0] - min_corner[0]) / 2,
                  (max_corner[1] - min_corner[1]) / 2,
                  (max_corner[2] - min_corner[2]) / 2};

    // test normal of the triangle
    // check if the projection of the triangle its normal lies in the interval
    // defined by the projecting of the box onto the same vector
    auto normal = vector_ab.cross(vector_ac);
    auto radius = extents[0] * Kokkos::abs(normal[0]) +
                  extents[1] * Kokkos::abs(normal[1]) +
                  extents[2] * Kokkos::abs(normal[2]);
    auto a_projected = a[0] * normal[0] + a[1] * normal[1] + a[2] * normal[2];
    if (Kokkos::abs(a_projected) > radius)
      return false;

    // Test crossproducts in a similar way as the triangle's normal above
    auto const vector_bc = c - b;

    // e_x x vector_ab = (0, -vector_ab[2],  vector_ab[1])
    {
      auto radius = extents[1] * Kokkos::abs(vector_ab[2]) +
                    extents[2] * Kokkos::abs(vector_ab[1]);
      auto xab_0 = -a[1] * vector_ab[2] + a[2] * vector_ab[1];
      auto xab_1 = -c[1] * vector_ab[2] + c[2] * vector_ab[1];
      if (Kokkos::fmin(xab_0, xab_1) > radius ||
          Kokkos::fmax(xab_0, xab_1) < -radius)
        return false;
    }
    {
      auto radius = extents[1] * Kokkos::abs(vector_ac[2]) +
                    extents[2] * Kokkos::abs(vector_ac[1]);
      auto xac_0 = -a[1] * vector_ac[2] + a[2] * vector_ac[1];
      auto xac_1 = -b[1] * vector_ac[2] + b[2] * vector_ac[1];
      if (Kokkos::fmin(xac_0, xac_1) > radius ||
          Kokkos::fmax(xac_0, xac_1) < -radius)
        return false;
    }
    {
      auto radius = extents[1] * Kokkos::abs(vector_bc[2]) +
                    extents[2] * Kokkos::abs(vector_bc[1]);
      auto xbc_0 = -a[1] * vector_bc[2] + a[2] * vector_bc[1];
      auto xbc_1 = -b[1] * vector_bc[2] + b[2] * vector_bc[1];
      if (Kokkos::fmin(xbc_0, xbc_1) > radius ||
          Kokkos::fmax(xbc_0, xbc_1) < -radius)
        return false;
    }

    // e_y x vector_ab = (vector_ab[2], 0, -vector_ab[0])
    {
      auto radius = extents[0] * Kokkos::abs(vector_ab[2]) +
                    extents[2] * Kokkos::abs(vector_ab[0]);
      auto yab_0 = a[0] * vector_ab[2] - a[2] * vector_ab[0];
      auto yab_1 = c[0] * vector_ab[2] - c[2] * vector_ab[0];
      if (Kokkos::fmin(yab_0, yab_1) > radius ||
          Kokkos::fmax(yab_0, yab_1) < -radius)
        return false;
    }
    {
      auto radius = extents[0] * Kokkos::abs(vector_ac[2]) +
                    extents[2] * Kokkos::abs(vector_ac[0]);
      auto yac_0 = a[0] * vector_ac[2] - a[2] * vector_ac[0];
      auto yac_1 = b[0] * vector_ac[2] - b[2] * vector_ac[0];
      if (Kokkos::fmin(yac_0, yac_1) > radius ||
          Kokkos::fmax(yac_0, yac_1) < -radius)
        return false;
    }
    {
      auto radius = extents[0] * Kokkos::abs(vector_bc[2]) +
                    extents[2] * Kokkos::abs(vector_bc[0]);
      auto ybc_0 = a[1] * vector_bc[2] - a[2] * vector_bc[0];
      auto ybc_1 = b[1] * vector_bc[2] - b[2] * vector_bc[0];
      if (Kokkos::fmin(ybc_0, ybc_1) > radius ||
          Kokkos::fmax(ybc_0, ybc_1) < -radius)
        return false;
    }

    // e_z x vector_ab = (-vector_ab[1], vector_ab[0], 0)
    {
      auto radius = extents[0] * Kokkos::abs(vector_ab[1]) +
                    extents[1] * Kokkos::abs(vector_ab[0]);
      auto zab_0 = -a[0] * vector_ab[1] + a[1] * vector_ab[0];
      auto zab_1 = -c[0] * vector_ab[1] + c[1] * vector_ab[0];
      if (Kokkos::fmin(zab_0, zab_1) > radius ||
          Kokkos::fmax(zab_0, zab_1) < -radius)
        return false;
    }
    {
      auto radius = extents[0] * Kokkos::abs(vector_ac[1]) +
                    extents[1] * Kokkos::abs(vector_ac[0]);
      auto xac_0 = -a[0] * vector_ac[1] + a[1] * vector_ac[0];
      auto xac_1 = -b[0] * vector_ac[1] + b[1] * vector_ac[0];
      if (Kokkos::fmin(xac_0, xac_1) > radius ||
          Kokkos::fmax(xac_0, xac_1) < -radius)
        return false;
    }
    {
      auto radius = extents[0] * Kokkos::abs(vector_bc[1]) +
                    extents[1] * Kokkos::abs(vector_bc[0]);
      auto zbc_0 = -a[0] * vector_bc[1] + a[1] * vector_bc[0];
      auto zbc_1 = -b[0] * vector_bc[1] + b[1] * vector_bc[0];
      if (Kokkos::fmin(zbc_0, zbc_1) > radius ||
          Kokkos::fmax(zbc_0, zbc_1) < -radius)
        return false;
    }
    return true;
  }
};

template <typename Triangle, typename Box>
struct intersects<TriangleTag, BoxTag, Triangle, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(Triangle const &triangle,
                                              Box const &box)
  {
    return intersects<BoxTag, TriangleTag, Box, Triangle>::apply(box, triangle);
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
          std::enable_if_t<GeometryTraits::is_point_v<Point> &&
                           GeometryTraits::is_box_v<Box>> * = nullptr>
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
