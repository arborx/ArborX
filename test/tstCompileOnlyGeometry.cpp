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

#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperPoint.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

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
  check_valid_geometry_traits(ArborX::Point{});
  check_valid_geometry_traits(ArborX::Box{});
  check_valid_geometry_traits(ArborX::Sphere{});

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

void test_point_ctad()
{
  using ArborX::ExperimentalHyperGeometry::Point;
  static_assert(std::is_same_v<decltype(Point{1}), Point<1, float>>);
  static_assert(std::is_same_v<decltype(Point{1.}), Point<1, double>>);
  static_assert(std::is_same_v<decltype(Point{1., 2}), Point<2, double>>);
  static_assert(std::is_same_v<decltype(Point{1, 2.}), Point<2, double>>);
  static_assert(std::is_same_v<decltype(Point{2, 2}), Point<2, float>>);
  static_assert(std::is_same_v<decltype(Point{2., 3.f, 2.}), Point<3, double>>);
  static_assert(
      std::is_same_v<decltype(Point{2.f, 3.f, 2.f}), Point<3, float>>);
  static_assert(
      std::is_same_v<decltype(Point<3, int>{2, 3, 2}), Point<3, int>>);
}

} // namespace ArborX::GeometryTraits
