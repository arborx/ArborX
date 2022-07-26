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

template <int DIM>
struct Morton32
{
  static int const dim = DIM;

  KOKKOS_FUNCTION auto operator()(BoxD<DIM> const &scene_bounding_box,
                                  PointD<DIM> p) const
  {
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p);
  }
  template <typename Geometry>
  KOKKOS_FUNCTION auto operator()(BoxD<DIM> const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    PointD<DIM> p;
    Details::centroid(geometry, p);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p);
  }
};

template <int DIM>
struct Morton64
{
  static int const dim = DIM;

  KOKKOS_FUNCTION auto operator()(BoxD<DIM> const &scene_bounding_box,
                                  PointD<DIM> p) const
  {
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p);
  }
  template <class Geometry>
  KOKKOS_FUNCTION auto operator()(BoxD<DIM> const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    PointD<DIM> p;
    Details::centroid(geometry, p);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p);
  }
};

} // namespace Experimental

namespace Details
{

template <class SpaceFillingCurve, class Geometry>
using SpaceFillingCurveProjectionArchetypeExpression =
    decltype(std::declval<SpaceFillingCurve const &>()(
        std::declval<BoxD<SpaceFillingCurve::dim> const &>(),
        std::declval<Geometry const &>()));

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
                         SpaceFillingCurve, PointD<SpaceFillingCurve::dim>>;
  static_assert(std::is_same<OrderValueType, unsigned int>::value ||
                    std::is_same<OrderValueType, unsigned long long>::value,
                "");
  static_assert(
      std::is_same<OrderValueType,
                   Kokkos::detected_t<
                       SpaceFillingCurveProjectionArchetypeExpression,
                       SpaceFillingCurve, BoxD<SpaceFillingCurve::dim>>>::value,
      "");
}

} // namespace Details
} // namespace ArborX

#endif
