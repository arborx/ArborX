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

struct NoTagSpecialization
{};
template <>
struct dimension<NoTagSpecialization>
{
  static constexpr int value = 3;
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

  // check_valid_geometry_traits(EmptyDimensionSpecialization{});

  // check_valid_geometry_traits(EmptyTagSpecialization{});

  // check_valid_geometry_traits(WrongTypeDimensionSpecialization{});

  // check_valid_geometry_traits(NegativeIntegerDimensionSpecialization{});

  // check_valid_geometry_traits(WrongTagSpecialization{});
}
} // namespace ArborX::GeometryTraits
