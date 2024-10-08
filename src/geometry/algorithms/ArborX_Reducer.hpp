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

#ifndef ARBORX_REDUCER_HPP
#define ARBORX_REDUCER_HPP

#include <ArborX_GeometryTraits.hpp>

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

} // namespace ArborX::Details

#endif
