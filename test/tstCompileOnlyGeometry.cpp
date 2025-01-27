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

#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Segment.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Triangle.hpp>

namespace ArborX::GeometryTraits
{

struct NoGeometryTraitsSpecialization
{};

struct NoDimensionSpecialization
{};
template <>
struct tag<NoDimensionSpecialization>
{
  using type = PointTag;
};
template <>
struct coordinate_type<NoDimensionSpecialization>
{
  using type = float;
};

struct NoCoordinateSpecialization
{};
template <>
struct tag<NoCoordinateSpecialization>
{
  using type = PointTag;
};
template <>
struct dimension<NoCoordinateSpecialization>
{
  static constexpr int value = 3;
};

struct NoTagSpecialization
{};
template <>
struct dimension<NoTagSpecialization>
{
  static constexpr int value = 3;
};
template <>
struct coordinate_type<NoTagSpecialization>
{
  using type = float;
};

struct EmptyDimensionSpecialization
{};
template <>
struct dimension<EmptyDimensionSpecialization>
{};
template <>
struct tag<EmptyDimensionSpecialization>
{
  using type = PointTag;
};
template <>
struct coordinate_type<EmptyDimensionSpecialization>
{
  using type = float;
};

struct EmptyTagSpecialization
{};
template <>
struct dimension<EmptyTagSpecialization>
{
  static constexpr int value = 3;
};
template <>
struct tag<EmptyTagSpecialization>
{};
template <>
struct coordinate_type<EmptyTagSpecialization>
{
  using type = float;
};

struct EmptyCoordinateSpecialization
{};
template <>
struct dimension<EmptyCoordinateSpecialization>
{
  static constexpr int value = 3;
};
template <>
struct tag<EmptyCoordinateSpecialization>
{
  using type = PointTag;
};
template <>
struct coordinate_type<EmptyCoordinateSpecialization>
{};

struct WrongTypeDimensionSpecialization
{};
template <>
struct dimension<WrongTypeDimensionSpecialization>
{
  static constexpr float value = 0.f;
};
template <>
struct tag<WrongTypeDimensionSpecialization>
{
  using type = PointTag;
};

struct NegativeIntegerDimensionSpecialization
{};
template <>
struct dimension<NegativeIntegerDimensionSpecialization>
{
  static constexpr int value = -1;
};
template <>
struct tag<NegativeIntegerDimensionSpecialization>
{
  using type = PointTag;
};

struct WrongTagSpecialization
{};
template <>
struct dimension<WrongTagSpecialization>
{
  static constexpr int value = 5;
};
struct DummyTag
{};
template <>
struct tag<WrongTagSpecialization>
{
  using type = DummyTag;
};

struct WrongCoordinateSpecialization
{};
template <>
struct dimension<WrongCoordinateSpecialization>
{
  static constexpr int value = 5;
};
template <>
struct tag<WrongCoordinateSpecialization>
{
  using type = PointTag;
};
template <>
struct coordinate_type<WrongCoordinateSpecialization>
{
  using type = DummyTag;
};

struct CorrectSpecialization
{};
template <>
struct dimension<CorrectSpecialization>
{
  static constexpr int value = 15;
};
template <>
struct tag<CorrectSpecialization>
{
  using type = SphereTag;
};
template <>
struct coordinate_type<CorrectSpecialization>
{
  using type = short;
};

void test_geometry_compile_only()
{
  check_valid_geometry_traits(ArborX::Point<3>{});
  check_valid_geometry_traits(ArborX::Box<3>{});
  check_valid_geometry_traits(ArborX::Sphere<3>{});

  check_valid_geometry_traits(CorrectSpecialization{});

  // Uncomment to see error messages

  // check_valid_geometry_traits(NoGeometryTraitsSpecialization{});

  // check_valid_geometry_traits(NoDimensionSpecialization{});

  // check_valid_geometry_traits(NoTagSpecialization{});

  // check_valid_geometry_traits(NoCoordinateSpecialization{});

  // check_valid_geometry_traits(EmptyDimensionSpecialization{});

  // check_valid_geometry_traits(EmptyTagSpecialization{});

  // check_valid_geometry_traits(EmptyCoordinateSpecialization{});

  // check_valid_geometry_traits(WrongTypeDimensionSpecialization{});

  // check_valid_geometry_traits(NegativeIntegerDimensionSpecialization{});

  // check_valid_geometry_traits(WrongTagSpecialization{});

  // check_valid_geometry_traits(WrongCoordinateSpecialization{});
}

void test_point_cv_compile_only()
{
  using Point = ArborX::Point<3>;
  namespace GT = ArborX::GeometryTraits;

  static_assert(GT::dimension_v<Point const> == 3);
  static_assert(std::is_same_v<GT::coordinate_type_t<Point const>, float>);
  static_assert(std::is_same_v<GT::tag_t<Point const>, GT::PointTag>);
  static_assert(GT::is_point_v<Point const>);
}

void test_point_ctad()
{
  using ArborX::Point;

  static_assert(std::is_same_v<decltype(Point{1}), Point<1, int>>);
  static_assert(std::is_same_v<decltype(Point{1.}), Point<1, double>>);
  static_assert(std::is_same_v<decltype(Point{{2.3, 3.3}}), Point<2, double>>);
  static_assert(std::is_same_v<decltype(Point{{1.f}}), Point<1, float>>);
  static_assert(
      std::is_same_v<decltype(Point{1.0L, 0.3L, 0.5L}), Point<3, long double>>);
  static_assert(
      std::is_same_v<decltype(Point{2.f, 3.f, 2.f}), Point<3, float>>);
  static_assert(
      std::is_same_v<decltype(Point<2, float>{1, 4}), Point<2, float>>);
  static_assert(
      std::is_same_v<decltype(Point<3, int>{2, 3, 2}), Point<3, int>>);
  static_assert(std::is_same_v<decltype(Point<3, double>{2.f, 3.f, 2.f}),
                               Point<3, double>>);
}

void test_segment_ctad()
{
  using ArborX::Point;
  using ArborX::Experimental::Segment;

  static_assert(std::is_same_v<decltype(Segment{{0.f, 2.f}, {2.f, 3.f}}),
                               Segment<2, float>>);
  static_assert(std::is_same_v<decltype(Segment{Point<2, double>{0.f, 2.f},
                                                Point<2, double>{2.f, 3.f}}),
                               Segment<2, double>>);
}

void test_box_ctad()
{
  using ArborX::Box;
  using ArborX::Point;

  static_assert(std::is_same_v<decltype(Box{{1.f, 3.f, 2.f}, {5.f, 7.f, 9.f}}),
                               Box<3, float>>);
  static_assert(std::is_same_v<decltype(Box{{1., 3., 2.}, {5., 7., 9.}}),
                               Box<3, double>>);
  static_assert(std::is_same_v<decltype(Box{Point{1.f, 2.f}, Point{3.f, 4.f}}),
                               Box<2, float>>);
}

void test_sphere_ctad()
{
  using ArborX::Point;
  using ArborX::Sphere;

  static_assert(
      std::is_same_v<decltype(Sphere{{0.f, 2.f, 5.f}, 2.f}), Sphere<3, float>>);
  static_assert(std::is_same_v<decltype(Sphere{Point{3., 4., 2.}, 6.}),
                               Sphere<3, double>>);
}

void test_triangle_ctad()
{
  using ArborX::Point;
  using ArborX::Triangle;

  static_assert(std::is_same_v<decltype(Triangle{{0, 2}, {3, 1}, {2, 5}}),
                               Triangle<2, int>>);
  static_assert(
      std::is_same_v<decltype(Triangle{{0.f, 2.f}, {3.f, 1.f}, {2.f, 5.f}}),
                     Triangle<2, float>>);
  static_assert(
      std::is_same_v<decltype(Triangle{Point{3., 4., 2.}, Point{2., 2., 2.},
                                       Point{6., 3., 5.}}),
                     Triangle<3, double>>);
}

} // namespace ArborX::GeometryTraits
