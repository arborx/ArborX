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
#ifndef ARBORX_GEOMETRY_BARYCENTRIC_COORDINATES_HPP
#define ARBORX_GEOMETRY_BARYCENTRIC_COORDINATES_HPP

#include <ArborX_GeometryTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX
{

namespace Details::Dispatch
{
template <typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct barycentric;
}

namespace Experimental
{
template <typename Geometry1, typename Geometry2>
KOKKOS_INLINE_FUNCTION auto barycentricCoordinates(Geometry1 const &geometry1,
                                                   Geometry2 const &geometry2)
{
  static_assert(GeometryTraits::dimension_v<Geometry1> ==
                GeometryTraits::dimension_v<Geometry2>);
  return Details::Dispatch::barycentric<GeometryTraits::tag_t<Geometry1>,
                                        GeometryTraits::tag_t<Geometry2>,
                                        Geometry1, Geometry2>::apply(geometry1,
                                                                     geometry2);
}
} // namespace Experimental

namespace Details::Dispatch
{

using namespace GeometryTraits;

template <typename Triangle, typename Point>
struct barycentric<TriangleTag, PointTag, Triangle, Point>
{
  KOKKOS_FUNCTION static constexpr auto apply(Triangle const &triangle,
                                              Point const &point)
  {
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;

    // ArborX should probably provide the interpolation function
    auto const a = triangle.a;
    auto const b = triangle.b;
    auto const c = triangle.c;

    // Find coefficients alpha and beta such that
    // x = a + alpha * (b - a) + beta * (c - a)
    //   = (1 - alpha - beta) * a + alpha * b + beta * c
    // recognizing the linear system
    // ((b - a) (c - a)) (alpha beta)^T = (x - a)
    Coordinate u[2] = {b[0] - a[0], b[1] - a[1]};
    Coordinate v[2] = {c[0] - a[0], c[1] - a[1]};
    Coordinate const det = v[1] * u[0] - v[0] * u[1];
    if (det == 0)
      Kokkos::abort("Degenerate triangles are not supported!");
    auto const inv_det = 1 / det;

    Coordinate alpha[2] = {v[1] * inv_det, -v[0] * inv_det};
    Coordinate beta[2] = {-u[1] * inv_det, u[0] * inv_det};

    auto alpha_coeff =
        alpha[0] * (point[0] - a[0]) + alpha[1] * (point[1] - a[1]);
    auto beta_coeff = beta[0] * (point[0] - a[0]) + beta[1] * (point[1] - a[1]);

    return Kokkos::Array<Coordinate, 3>{1 - alpha_coeff - beta_coeff,
                                        alpha_coeff, beta_coeff};
  }
};

template <typename Tetrahedron, typename Point>
struct barycentric<TetrahedronTag, PointTag, Tetrahedron, Point>
{
  template <typename Coordinate>
  using Vector = ::ArborX::Details::Vector<3, Coordinate>;

  template <typename Coordinate>
  KOKKOS_FUNCTION static constexpr auto
  triple_product(Vector<Coordinate> const &u, Vector<Coordinate> const &v,
                 Vector<Coordinate> const &w)
  {
    return u.dot(v.cross(w));
  }

  KOKKOS_FUNCTION static constexpr auto apply(Tetrahedron const &tet,
                                              Point const &point)
  {
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;

    auto ap = point - tet.a;
    auto bp = point - tet.b;

    auto ab = tet.b - tet.a;
    auto ac = tet.c - tet.a;
    auto ad = tet.d - tet.a;

    auto bc = tet.c - tet.b;
    auto bd = tet.d - tet.b;

    auto denom = 1 / triple_product(ab, ac, ad);
    return Kokkos::Array<Coordinate, 4>{
        triple_product(bp, bd, bc) * denom, triple_product(ap, ac, ad) * denom,
        triple_product(ap, ad, ab) * denom, triple_product(ap, ab, ac) * denom};
  }
};

} // namespace Details::Dispatch

} // namespace ArborX

#endif
