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

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsMortonCode.hpp>

#include <Kokkos_DetectionIdiom.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Experimental
{

struct Morton32
{
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box, Point p) const
  {
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p[0], p[1], p[2]);
  }
  template <typename Geometry>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    Point p;
    Details::centroid(geometry, p);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p[0], p[1], p[2]);
  }
};

struct Morton64
{
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box, Point p) const
  {
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p[0], p[1], p[2]);
  }
  template <class Geometry>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    Point p;
    Details::centroid(geometry, p);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p[0], p[1], p[2]);
  }
};

} // namespace Experimental

namespace Details
{

template <class SpaceFillingCurve, class Geometry>
using SpaceFillingCurveProjectionArchetypeExpression =
    decltype(std::declval<SpaceFillingCurve const &>()(
        std::declval<Box const &>(), std::declval<Geometry const &>()));

template <class SpaceFillingCurve>
void check_valid_space_filling_curve(SpaceFillingCurve const &)
{
  static_assert(
      Kokkos::is_detected<SpaceFillingCurveProjectionArchetypeExpression,
                          SpaceFillingCurve, Point>::value,
      "");
  static_assert(
      Kokkos::is_detected<SpaceFillingCurveProjectionArchetypeExpression,
                          SpaceFillingCurve, Box>::value,
      "");
  using OrderValueType =
      Kokkos::detected_t<SpaceFillingCurveProjectionArchetypeExpression,
                         SpaceFillingCurve, Point>;
  static_assert(std::is_same<OrderValueType, unsigned int>::value ||
                    std::is_same<OrderValueType, unsigned long long>::value,
                "");
  static_assert(
      std::is_same<
          OrderValueType,
          Kokkos::detected_t<SpaceFillingCurveProjectionArchetypeExpression,
                             SpaceFillingCurve, Box>>::value,
      "");
}

} // namespace Details
} // namespace ArborX

#endif
