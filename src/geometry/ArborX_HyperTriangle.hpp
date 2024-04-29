/****************************************************************************
 * Copyright (c) 2017-2024 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_TRIANGLE_HPP
#define ARBORX_TRIANGLE_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_MathematicalFunctions.hpp> // fmax, fmin

namespace ArborX::ExperimentalHyperGeometry
{
// need to add a protection that
// the points are not on the same line.
template <int DIM, class Coordinate = float>
struct Triangle
{
  ExperimentalHyperGeometry::Point<DIM, Coordinate> a;
  ExperimentalHyperGeometry::Point<DIM, Coordinate> b;
  ExperimentalHyperGeometry::Point<DIM, Coordinate> c;
};

template <int DIM, class Coordinate>
Triangle(ExperimentalHyperGeometry::Point<DIM, Coordinate>,
         ExperimentalHyperGeometry::Point<DIM, Coordinate>,
         ExperimentalHyperGeometry::Point<DIM, Coordinate>)
    -> Triangle<DIM, Coordinate>;

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  using type = TriangleTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  using type = Coordinate;
};

namespace ArborX::Details::Dispatch
{
using GeometryTraits::BoxTag;
using GeometryTraits::PointTag;
using GeometryTraits::TriangleTag;

// distance point-triangle
template <typename Point, typename Triangle>
struct distance<PointTag, TriangleTag, Point, Triangle>
{
  static constexpr int DIM = GeometryTraits::dimension_v<Point>;
  using Coordinate = GeometryTraits::coordinate_type_t<Triangle>;

  static_assert(DIM == 2 || DIM == 3);

  struct Vector : private Point
  {
    using Point::Point;
    using Point::operator[];
    KOKKOS_FUNCTION Vector(Point const &a, Point const &b)
    {
      for (int d = 0; d < DIM; ++d)
        (*this)[d] = b[d] - a[d];
    }
  };

  KOKKOS_FUNCTION static auto dot_product(Vector const &v, Vector const &w)
  {
    auto r = v[0] * w[0];
    for (int d = 1; d < DIM; ++d)
      r += v[d] * w[d];
    return r;
  }
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

    Vector ab(a, b);
    Vector ac(a, c);
    Vector ap(a, p);

    auto const d1 = dot_product(ab, ap);
    auto const d2 = dot_product(ac, ap);
    if (d1 <= 0 && d2 <= 0) // zone 1
      return a;

    Vector bp(b, p);
    auto const d3 = dot_product(ab, bp);
    auto const d4 = dot_product(ac, bp);
    if (d3 >= 0 && d4 <= d3) // zone 2
      return b;

    Vector cp(c, p);
    auto const d5 = dot_product(ab, cp);
    auto const d6 = dot_product(ac, cp);
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
    Point vector_ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    Point vector_ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    Point extents{(max_corner[0] - min_corner[0]) / 2,
                  (max_corner[1] - min_corner[1]) / 2,
                  (max_corner[2] - min_corner[2]) / 2};

    // test normal of the triangle
    // check if the projection of the triangle its normal lies in the interval
    // defined by the projecting of the box onto the same vector
    Point normal{{vector_ab[1] * vector_ac[2] - vector_ab[2] * vector_ac[1],
                  vector_ab[2] * vector_ac[0] - vector_ab[0] * vector_ac[2],
                  vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0]}};
    auto radius = extents[0] * Kokkos::abs(normal[0]) +
                  extents[1] * Kokkos::abs(normal[1]) +
                  extents[2] * Kokkos::abs(normal[2]);
    auto a_projected = a[0] * normal[0] + a[1] * normal[1] + a[2] * normal[2];
    if (Kokkos::abs(a_projected) > radius)
      return false;

    // Test crossproducts in a similar way as the triangle's normal above
    Point vector_bc{c[0] - b[0], c[1] - b[1], c[2] - b[2]};

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

} // namespace ArborX::Details::Dispatch

#endif
