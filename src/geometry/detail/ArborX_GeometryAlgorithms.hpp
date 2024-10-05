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
#ifndef ARBORX_ALGORITHMS_HPP
#define ARBORX_ALGORITHMS_HPP

#include "ArborX_GeometryCentroid.hpp"
#include "ArborX_GeometryConvert.hpp"
#include "ArborX_GeometryDistance.hpp"
#include "ArborX_GeometryEquals.hpp"
#include "ArborX_GeometryExpand.hpp"
#include "ArborX_GeometryIntersects.hpp"
#include "ArborX_GeometryValid.hpp"
#include <ArborX_GeometryTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <class Geometry, class Space = Kokkos::HostSpace>
struct GeometryReducer
{
  static_assert(GeometryTraits::is_valid_geometry<Geometry>);

  using reducer = GeometryReducer<Geometry, Space>;
  using value_type = std::remove_cv_t<Geometry>;
  static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

  using result_view_type = Kokkos::View<value_type, Space>;

private:
  result_view_type _value;
  bool _references_scalar;

public:
  KOKKOS_FUNCTION
  GeometryReducer(value_type &value)
      : _value(&value)
      , _references_scalar(true)
  {}

  KOKKOS_FUNCTION
  GeometryReducer(result_view_type const &value)
      : _value(value)
      , _references_scalar(false)
  {}

  KOKKOS_FUNCTION
  void join(value_type &dest, value_type const &src) const
  {
    expand(dest, src);
  }

  KOKKOS_FUNCTION
  void init(value_type &value) const { value = {}; }

  KOKKOS_FUNCTION
  value_type &reference() const { return *_value.data(); }

  KOKKOS_FUNCTION
  result_view_type view() const { return _value; }

  KOKKOS_FUNCTION
  bool references_scalar() const { return _references_scalar; }
};

// transformation that maps the unit cube into a new axis-aligned box
// NOTE safe to perform in-place
template <typename Point, typename Box,
          std::enable_if_t<GeometryTraits::is_point_v<Point> &&
                           GeometryTraits::is_box_v<Box>> * = nullptr>
KOKKOS_FUNCTION void translateAndScale(Point const &in, Point &out,
                                       Box const &ref)
{
  static_assert(GeometryTraits::dimension_v<Point> ==
                GeometryTraits::dimension_v<Box>);
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  for (int d = 0; d < DIM; ++d)
  {
    auto const a = ref.minCorner()[d];
    auto const b = ref.maxCorner()[d];
    out[d] = (a != b ? (in[d] - a) / (b - a) : 0);
  }
}

} // namespace ArborX::Details

#endif
