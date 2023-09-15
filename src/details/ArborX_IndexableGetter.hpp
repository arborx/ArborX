/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_INDEXABLE_GETTER_HPP
#define ARBORX_INDEXABLE_GETTER_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsNode.hpp> // PairIndexVolume
#include <ArborX_GeometryTraits.hpp>

namespace ArborX::Details
{

struct DefaultIndexableGetter
{
  KOKKOS_DEFAULTED_FUNCTION
  DefaultIndexableGetter() = default;

  template <typename Geometry, typename Enable = std::enable_if_t<
                                   GeometryTraits::is_point<Geometry>{} ||
                                   GeometryTraits::is_box<Geometry>{}>>
  KOKKOS_FUNCTION auto const &operator()(Geometry const &geometry) const
  {
    return geometry;
  }

  template <typename Geometry>
  KOKKOS_FUNCTION Geometry const &
  operator()(PairIndexVolume<Geometry> const &pair) const
  {
    return pair.bounding_volume;
  }

  template <typename Geometry>
  KOKKOS_FUNCTION Geometry operator()(PairIndexVolume<Geometry> &&pair) const
  {
    return pair.bounding_volume;
  }
};

template <typename Values, typename IndexableGetter>
struct Indexables
{
  Values _values;
  IndexableGetter _indexable_getter;

  using memory_space = typename Values::memory_space;

  KOKKOS_FUNCTION decltype(auto) operator()(int i) const
  {
    return _indexable_getter(_values(i));
  }

  KOKKOS_FUNCTION auto size() const { return _values.size(); }
};

} // namespace ArborX::Details

#endif
