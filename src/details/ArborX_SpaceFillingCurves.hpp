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

#ifndef ARBORX_SPACE_FILLING_CURVES_HPP
#define ARBORX_SPACE_FILLING_CURVES_HPP

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsMortonCode.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_HyperPoint.hpp>

#include <Kokkos_DetectionIdiom.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Experimental
{

struct Morton32
{
  template <typename Box, typename Point,
            std::enable_if_t<GeometryTraits::is_box<Box>{} &&
                             GeometryTraits::is_point<Point>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box, Point p) const
  {
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p);
  }
  template <typename Box, typename Geometry,
            std::enable_if_t<GeometryTraits::is_box<Box>{} &&
                             !GeometryTraits::is_point<Geometry>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    auto p = Details::returnCentroid(geometry);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p);
  }
};

struct Morton64
{
  template <typename Box, typename Point,
            std::enable_if_t<GeometryTraits::is_box<Box>{} &&
                             GeometryTraits::is_point<Point>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box, Point p) const
  {
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p);
  }
  template <typename Box, class Geometry,
            std::enable_if_t<GeometryTraits::is_box<Box>{} &&
                             !GeometryTraits::is_point<Geometry>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    auto p = Details::returnCentroid(geometry);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p);
  }
};

} // namespace Experimental

namespace Details
{

template <class SpaceFillingCurve, class Box, class Geometry>
using SpaceFillingCurveProjectionArchetypeExpression =
    std::invoke_result_t<SpaceFillingCurve, Box, Geometry>;

template <int DIM, class SpaceFillingCurve>
void check_valid_space_filling_curve(SpaceFillingCurve const &)
{
  using Point = ExperimentalHyperGeometry::Point<DIM>;
  using Box = ExperimentalHyperGeometry::Box<DIM>;

  static_assert(
      Kokkos::is_detected<SpaceFillingCurveProjectionArchetypeExpression,
                          SpaceFillingCurve, Box, Point>::value);
  static_assert(
      Kokkos::is_detected<SpaceFillingCurveProjectionArchetypeExpression,
                          SpaceFillingCurve, Box, Box>::value);
  using OrderValueType =
      Kokkos::detected_t<SpaceFillingCurveProjectionArchetypeExpression,
                         SpaceFillingCurve, Box, Point>;
  static_assert(std::is_same<OrderValueType, unsigned int>::value ||
                std::is_same<OrderValueType, unsigned long long>::value);
  static_assert(
      std::is_same<
          OrderValueType,
          Kokkos::detected_t<SpaceFillingCurveProjectionArchetypeExpression,
                             SpaceFillingCurve, Box, Box>>::value);
}

} // namespace Details
} // namespace ArborX

#endif
