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
#ifndef ARBORX_OBB_HPP
#define ARBORX_OBB_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsContainers.hpp> // StaticVector
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsSymmetricSVD.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_DetailsVector.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX::Experimental
{

namespace
{
template <typename T>
using UnmanagedViewWrapper =
    Kokkos::View<T, Kokkos::AnonymousSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
}

template <int DIM, typename Coordinate = float>
struct OBB
{
  // Ortho-normal transformation matrix
  // Normalized eigenvectors are stores in columns.
  // The inverse is the transpose.
  Coordinate _matrix[DIM * DIM];

  // Box in the new coordinate system
  ExperimentalHyperGeometry::Box<DIM, Coordinate> _box;

  KOKKOS_DEFAULTED_FUNCTION OBB() = default;

  template <typename View, typename = std::enable_if_t<Kokkos::is_view_v<View>>>
  KOKKOS_FUNCTION OBB(View const &points)
  {
    static_assert(View::rank == 1);

    using Point = typename View::value_type;
    static_assert(GeometryTraits::is_point_v<Point>);
    static_assert(GeometryTraits::dimension_v<Point> == DIM);
    static_assert(
        std::is_same_v<GeometryTraits::coordinate_type_t<Point>, Coordinate>);

    int const n = points.size();
    KOKKOS_ASSERT(n > 0);

    auto mat = matrix();

    if (n == 1)
    {
      // Set matrix to identity
      for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j)
          mat(i, j) = (i == j ? 1 : 0);

      Details::expand(_box, points[0]);
      return;
    }

    // Compute covariance matrix
    Coordinate cov_data[DIM * DIM];
    UnmanagedViewWrapper<Coordinate[DIM][DIM]> cov(cov_data);
    {
      // There's a way to optimize computing covariance in a single loop,
      // however, it is likely less precise
      Coordinate mean[DIM];
      for (int i = 0; i < DIM; ++i)
      {
        mean[i] = 0;
        for (int k = 0; k < n; ++k)
          mean[i] += points(k)[i];
        mean[i] /= n;
      }
      for (int i = 0; i < DIM; ++i)
        for (int j = i; j < DIM; ++j)
        {
          cov(i, j) = 0;
          for (int k = 0; k < n; ++k)
            cov(i, j) += (points(k)[i] - mean[i]) * (points(k)[j] - mean[j]);
          cov(i, j) /= n;
        }

      // FIXME: not sure we need to do this (depends on SymSVD implementation)
      for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < i; ++j)
          cov(i, j) = cov(j, i);
    }

    // Find the orthonormalized eigenvectors and store them
    Coordinate d_data[DIM];
    UnmanagedViewWrapper<Coordinate[DIM]> D(d_data);

    Interpolation::Details::symmetricSVDKernel(cov, D, mat);

    // Find extents by projecting points
    for (int i = 0; i < n; ++i)
      Details::expand(_box, transform(points(i)));
  }

  template <int N>
  KOKKOS_FUNCTION
  OBB(ExperimentalHyperGeometry::Point<DIM, Coordinate> const (&points)[N])
      : OBB(UnmanagedViewWrapper<
            const ExperimentalHyperGeometry::Point<DIM, Coordinate>[N]>(points))
  {}

  KOKKOS_FUNCTION auto matrix()
  {
    return UnmanagedViewWrapper<Coordinate[DIM][DIM]>(_matrix);
  }

  KOKKOS_FUNCTION auto matrix() const
  {
    return UnmanagedViewWrapper<const Coordinate[DIM][DIM]>(_matrix);
  }

  template <typename Point>
  KOKKOS_FUNCTION auto transform(Point const &point) const
  {
    static_assert(GeometryTraits::is_point_v<Point>);
    static_assert(GeometryTraits::dimension_v<Point> == DIM);

    auto const &mat = matrix();

    // Use matrix transpose
    Point p;
    for (int row = 0; row < DIM; ++row)
    {
      p[row] = 0;
      for (int col = 0; col < DIM; ++col)
        p[row] += mat(col, row) * point[col];
    }
    return p;
  }

  template <typename Point>
  KOKKOS_FUNCTION auto transform_back(Point const &point) const
  {
    static_assert(GeometryTraits::is_point_v<Point>);
    static_assert(GeometryTraits::dimension_v<Point> == DIM);

    auto const &mat = matrix();

    Point p;
    for (int row = 0; row < DIM; ++row)
    {
      p[row] = 0;
      for (int col = 0; col < DIM; ++col)
        p[row] += mat(row, col) * point[col];
    }
    return p;
  }

  KOKKOS_FUNCTION auto corners() const
  {
    static_assert(DIM == 2 || DIM == 3);

    KOKKOS_ASSERT(Details::isValid(*this));

    using Point = ExperimentalHyperGeometry::Point<DIM, Coordinate>;

    Coordinate const xs[2] = {_box.minCorner()[0], _box.maxCorner()[0]};
    Coordinate const ys[2] = {_box.minCorner()[1], _box.maxCorner()[1]};
    int const num_unique_xs = (xs[0] != xs[1]) + 1;
    int const num_unique_ys = (ys[0] != ys[1]) + 1;
    if constexpr (DIM == 2)
    {
      Details::StaticVector<Point, 4> points;
      for (int i = 0; i < num_unique_xs; ++i)
        for (int j = 0; j < num_unique_ys; ++j)
          points.emplaceBack(transform_back(Point{xs[i], ys[j]}));
      return points;
    }
    else
    {
      Coordinate const zs[2] = {_box.minCorner()[2], _box.maxCorner()[2]};
      int const num_unique_zs = (zs[0] != zs[1]) + 1;
      Details::StaticVector<Point, 8> points;
      for (int i = 0; i < num_unique_xs; ++i)
        for (int j = 0; j < num_unique_ys; ++j)
          for (int k = 0; k < num_unique_zs; ++k)
            points.emplaceBack(transform_back(Point{xs[i], ys[j], zs[k]}));
      return points;
    }
  }

  // FIXME: only necessary to not modify the tests
  KOKKOS_FUNCTION explicit operator Box() const
  {
    Box box;
    Details::expand(box, *this);
    return box;
  }
};

} // namespace ArborX::Experimental

template <int DIM, typename Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::OBB<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, typename Coordinate>
struct ArborX::GeometryTraits::tag<ArborX::Experimental::OBB<DIM, Coordinate>>
{
  using type = OBBTag;
};
template <int DIM, typename Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::OBB<DIM, Coordinate>>
{
  using type = Coordinate;
};

namespace ArborX::Details::Dispatch
{

template <typename OBB>
struct equals<OBBTag, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(OBB const &l, OBB const &r)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        if (l.matrix()(i, j) != r.matrix()(i, j))
          return false;
    return Details::equals(l._box, r._box);
  }
};

template <typename OBB>
struct isValid<OBBTag, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(OBB const &obb)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;

    // Slight modification on isValid(Box) in that Box{p, p} would be valid
    // here
    auto const &b = obb._box;
    for (int d = 0; d < DIM; ++d)
    {
      auto const r_d = b.maxCorner()[d] - b.minCorner()[d];
      if (!Kokkos::isfinite(r_d) || r_d < 0)
        return false;
    }
    return true;
  }
};

template <typename OBB, typename Point>
struct expand<OBBTag, PointTag, OBB, Point>
{
  KOKKOS_FUNCTION static void apply(OBB &obb, Point const &point)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;
    using Coordinate = GeometryTraits::coordinate_type_t<OBB>;

    using HyperPoint = ExperimentalHyperGeometry::Point<DIM, Coordinate>;
    auto hyper_point = Kokkos::bit_cast<HyperPoint>(point);

    if (!Details::isValid(obb))
    {
      obb = OBB({hyper_point});
      return;
    }

    auto const corners = obb.corners();
    int const num_corners = corners.size();

    Details::StaticVector<HyperPoint, corners.capacity() + 1> points;
    for (int i = 0; i < num_corners; ++i)
      points.emplaceBack(corners[i]);
    points.emplaceBack(hyper_point);

    obb = OBB(Kokkos::View<HyperPoint *, Kokkos::AnonymousSpace,
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        points.data(), points.size()));
  }
};

template <typename OBB, typename Box>
struct expand<OBBTag, BoxTag, OBB, Box>
{
  KOKKOS_FUNCTION static void apply(OBB &obb, Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<OBB>;

    OBB other;
    // Set matrix to identity
    for (int i = 0; i < DIM; ++i)
      for (int j = 0; j < DIM; ++j)
        other.matrix()(i, j) = (i == j ? 1 : 0);
    // FIXME: doing expand instead of _box = box to accomodate both regular and
    // hyper-dimensional box
    Details::expand(other._box, box);

    if (!Details::isValid(obb))
      obb = other;
    else
      Details::expand(obb, other);
  }
};

template <typename Box, typename OBB>
struct expand<BoxTag, OBBTag, Box, OBB>
{
  KOKKOS_FUNCTION static void apply(Box &box, OBB const &obb)
  {
    if (!Details::isValid(obb))
      return;

    auto const corners = obb.corners();
    int const num_corners = corners.size();
    for (int i = 0; i < num_corners; ++i)
      Details::expand(box, corners[i]);
  }
};

template <typename OBB1, typename OBB2>
struct expand<OBBTag, OBBTag, OBB1, OBB2>
{
  KOKKOS_FUNCTION static void apply(OBB1 &obb, OBB2 const &other)
  {
    KOKKOS_ASSERT(Details::isValid(other));

    if (!Details::isValid(obb))
    {
      obb = other;
      return;
    }

    auto corners1 = obb.corners();
    auto corners2 = other.corners();
    int const num_corners1 = corners1.size();
    int const num_corners2 = corners2.size();

    using Point = typename decltype(corners1)::value_type;

    Details::StaticVector<Point, corners1.capacity() + corners2.capacity()>
        points;
    for (int i = 0; i < num_corners1; ++i)
      points.emplaceBack(corners1[i]);
    for (int i = 0; i < num_corners2; ++i)
      points.emplaceBack(corners2[i]);

    obb = OBB1(Kokkos::View<Point *, Kokkos::AnonymousSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        points.data(), points.size()));
  }
};

template <typename OBB>
struct centroid<OBBTag, OBB>
{
  KOKKOS_FUNCTION static auto apply(OBB const &obb)
  {
    return obb.transform_back(Details::returnCentroid(obb._box));
  }
};

template <typename Point, typename OBB>
struct intersects<PointTag, OBBTag, Point, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              OBB const &obb)
  {
    return Details::intersects(obb.transform(point), obb._box);
  }
};

template <typename Sphere, typename OBB>
struct intersects<SphereTag, OBBTag, Sphere, OBB>
{
  KOKKOS_FUNCTION static constexpr bool apply(Sphere const &sphere,
                                              OBB const &obb)
  {
    return Details::intersects(
        Sphere{obb.transform(sphere.centroid()), sphere.radius()}, obb._box);
  }
};

template <typename Point, typename OBB>
struct distance<PointTag, OBBTag, Point, OBB>
{
  KOKKOS_FUNCTION static auto apply(Point const &point, OBB const &obb)
  {
    return Details::distance(obb.transform(point), obb._box);
  }
};

} // namespace ArborX::Details::Dispatch

#endif
