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

#include <ArborX_GeometryTraits.hpp>
#include <algorithms/ArborX_Centroid.hpp>
#include <algorithms/ArborX_TranslateAndScale.hpp>
#include <detail/ArborX_MortonCode.hpp>
#include <misc/ArborX_SortUtils.hpp>

#include <Kokkos_DetectionIdiom.hpp>
#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace ArborX
{
namespace Experimental
{

struct Morton32
{
  template <typename Box, typename Geometry>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    static_assert(GeometryTraits::is_box_v<Box>);
    using Details::returnCentroid;
    auto p = returnCentroid(geometry);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton32(p);
  }
};

struct Morton64
{
  template <typename Box, typename Geometry>
  KOKKOS_FUNCTION auto operator()(Box const &scene_bounding_box,
                                  Geometry const &geometry) const
  {
    static_assert(GeometryTraits::is_box_v<Box>);
    using Details::returnCentroid;
    auto p = returnCentroid(geometry);
    Details::translateAndScale(p, p, scene_bounding_box);
    return Details::morton64(p);
  }
};

} // namespace Experimental

namespace Details
{

template <typename ExecutionSpace, typename Values, typename SpaceFillingCurve,
          typename Box, typename LinearOrdering>
inline void
projectOntoSpaceFillingCurve(ExecutionSpace const &space, Values const &values,
                             SpaceFillingCurve const &curve,
                             Box const &scene_bounding_box,
                             LinearOrdering &linear_ordering_indices)
{
  using Point = std::decay_t<decltype(returnCentroid(values(0)))>;
  static_assert(GeometryTraits::is_point_v<Point>);
  static_assert(GeometryTraits::is_box_v<Box>);
  ARBORX_ASSERT(linear_ordering_indices.size() == values.size());
  static_assert(std::is_same_v<typename LinearOrdering::value_type,
                               decltype(curve(scene_bounding_box, values(0)))>);

  Kokkos::parallel_for(
      "ArborX::SpaceFillingCurve::project_onto_space_filling_curve",
      Kokkos::RangePolicy(space, 0, values.size()), KOKKOS_LAMBDA(int i) {
        linear_ordering_indices(i) = curve(scene_bounding_box, values(i));
      });
}

template <typename ExecutionSpace, typename Values, typename SpaceFillingCurve,
          typename Box>
inline auto computeSpaceFillingCurvePermutation(ExecutionSpace const &space,
                                                Values const &values,
                                                SpaceFillingCurve const &curve,
                                                Box const &scene_bounding_box)
{
  using Point = std::decay_t<decltype(returnCentroid(values(0)))>;
  using LinearOrderingValueType =
      std::invoke_result_t<SpaceFillingCurve, Box, Point>;
  Kokkos::View<LinearOrderingValueType *, typename Values::memory_space>
      linear_ordering_indices(
          Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                             "ArborX::SpaceFillingCurve::linear_ordering"),
          values.size());
  projectOntoSpaceFillingCurve(space, values, curve, scene_bounding_box,
                               linear_ordering_indices);
  return sortObjects(space, linear_ordering_indices);
}

template <int DIM, class SpaceFillingCurve>
void check_valid_space_filling_curve(SpaceFillingCurve const &)
{
  using Point = Point<DIM>;
  using Box = Box<DIM>;

  static_assert(std::is_invocable_v<SpaceFillingCurve const &, Box, Point>);
  static_assert(std::is_invocable_v<SpaceFillingCurve const &, Box, Box>);

  using OrderValueType =
      std::invoke_result_t<SpaceFillingCurve const &, Box, Point>;
  static_assert(std::is_same_v<OrderValueType, unsigned int> ||
                std::is_same_v<OrderValueType, unsigned long long>);
  static_assert(std::is_same_v<
                OrderValueType,
                std::invoke_result_t<SpaceFillingCurve const &, Box, Box>>);
}

} // namespace Details
} // namespace ArborX

#endif
