/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_GEOMETRY_INTERSECTS_HPP
#define ARBORX_DETAILS_GEOMETRY_INTERSECTS_HPP

#include "ArborX_GeometryDistance.hpp"
#include "ArborX_GeometryExpand.hpp"
#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace ArborX::Details
{
namespace Dispatch
{
template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct intersects;
}

template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION constexpr bool intersects(Geometry1 const &geometry1,
                                                 Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Dispatch::intersects<GeometryTraits::tag_t<Geometry1>,
                              GeometryTraits::tag_t<Geometry2>, Geometry1,
                              Geometry2>::apply(geometry1, geometry2);
}

namespace Dispatch
{

using namespace GeometryTraits;

// check if two axis-aligned bounding boxes intersect
template <typename Box1, typename Box2>
struct intersects<BoxTag, BoxTag, Box1, Box2>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box1 const &box,
                                              Box2 const &other)
  {
    constexpr int DIM = dimension_v<Box1>;
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
                                              Box const &box)
  {
    constexpr int DIM = dimension_v<Point>;
    for (int d = 0; d < DIM; ++d)
      if (point[d] > box.maxCorner()[d] || point[d] < box.minCorner()[d])
        return false;
    return true;
  }
};
template <typename Box, typename Point>
struct intersects<BoxTag, PointTag, Box, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &box,
                                              Point const &point)
  {
    return Details::intersects(point, box);
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
    constexpr int DIM = dimension_v<Point>;
    static_assert(DIM == 2);

    auto const &a = triangle.a;
    auto const &b = triangle.b;
    auto const &c = triangle.c;

    using Float = coordinate_type_t<Point>;

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
    Details::expand(bounding_box, triangle);
    if (!Details::intersects(bounding_box, box))
      return false;

    // shift box and triangle so that the box's center is at the origin to
    // simplify the following checks.
    constexpr int DIM = dimension_v<Triangle>;
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

template <typename KDOP, typename Box>
struct intersects<KDOPTag, BoxTag, KDOP, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(KDOP const &kdop, Box const &box)
  {
    KDOP other{};
    Details::expand(other, box);
    return Details::intersects(kdop, other);
  }
};

template <typename Box, typename KDOP>
struct intersects<BoxTag, KDOPTag, Box, KDOP>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &box, KDOP const &kdop)
  {
    return Details::intersects(kdop, box);
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

template <typename Point, typename KDOP>
struct intersects<PointTag, KDOPTag, Point, KDOP>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              KDOP const &kdop)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    constexpr int n_directions = KDOP::n_directions;
    for (int i = 0; i < n_directions; ++i)
    {
      auto const &dir = kdop.directions()[i];
      auto proj_i = point[0] * dir[0];
      for (int d = 1; d < DIM; ++d)
        proj_i += point[d] * dir[d];

      if (proj_i < kdop._min_values[i] || proj_i > kdop._max_values[i])
        return false;
    }
    return true;
  }
};

template <typename KDOP, typename Point>
struct intersects<KDOPTag, PointTag, KDOP, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(KDOP const &kdop,
                                              Point const &point)
  {
    return Details::intersects(point, kdop);
  }
};

template <typename KDOP1, typename KDOP2>
struct intersects<KDOPTag, KDOPTag, KDOP1, KDOP2>
{
  KOKKOS_FUNCTION static constexpr bool apply(KDOP1 const &kdop,
                                              KDOP2 const &other)
  {
    constexpr int n_directions = KDOP1::n_directions;
    static_assert(KDOP2::n_directions == n_directions);
    for (int i = 0; i < kdop.n_directions; ++i)
    {
      if (other._max_values[i] < kdop._min_values[i] ||
          other._min_values[i] > kdop._max_values[i])
      {
        return false;
      }
    }
    return true;
  }
};

template <typename Point, typename Tetrahedron>
struct intersects<PointTag, TetrahedronTag, Point, Tetrahedron>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              Tetrahedron const &tet)
  {
    static_assert(GeometryTraits::dimension_v<Point> == 3);

    constexpr int N = 4;
    Kokkos::Array<decltype(tet.a), N> v = {tet.a, tet.b, tet.c, tet.d};

    // For every plane check that the vertex lies within the same halfspace as
    // the other tetrahedron vertex. This is a simple but possibly not very
    // efficient algorithm.
    for (int j = 0; j < N; ++j)
    {
      auto normal = (v[(j + 1) % N] - v[j]).cross(v[(j + 2) % N] - v[j]);

      bool same_half_space =
          (normal.dot(v[(j + 3) % N] - v[j]) * normal.dot(point - v[j]) >= 0);
      if (!same_half_space)
        return false;
    }
    return true;
  }
};

} // namespace Dispatch

} // namespace ArborX::Details

#endif
